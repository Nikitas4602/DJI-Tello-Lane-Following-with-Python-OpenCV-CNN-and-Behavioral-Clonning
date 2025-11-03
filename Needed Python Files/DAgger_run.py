# DAgger_run.py — stream like inference, autonomous YAW, expert corrections like collect_data,
# keep all policy snapshots (no deletions), optional flip OFF by default.

import csv, time, cv2, ctypes, torch, pandas as pd, numpy as np
from pathlib import Path

from model import SmallCNN
from utils import (
    safe_tello, open_cv_stream, read_frame, visualize,
    safe_cleanup, robust_takeoff, robust_land
)
from config import (
    # device
    DEVICE,
    # stream
    TELLO_STREAM_URL, STREAM_WARMUP_SEC,
    # control gains
    INVERT_YAW, STEER_DEADBAND, YAW_GAIN, MAX_YAW_RC,
    FORWARD_SPEED,
    # altitude hold
    ALT_TARGET_CM, ALT_KP, ALT_UD_LIMIT,
)

# ================= MANUAL POLICY SELECTION =================
# Εδώ βάζεις εσύ το path του μοντέλου που θέλεις να φορτώσει το DAgger
#MODEL_PATH = r"C:\Users\nikpa\PycharmProjects\tello_line_following\data\dagger_20251103-105042.pt"
MODEL_PATH = r"C:\Users\nikpa\PycharmProjects\tello_line_following\models\policy_latest.pt"
print(f"[DAgger] Using manual model path: {MODEL_PATH}")
# ===========================================================

# --- Paths ---
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"    # απόλυτο ./models
DATA_ROOT = ROOT / "data"       # ρίζα για τα sessions DAgger

# ---- Optional flip (OFF by default) ----
USE_FLIP = True  # βάλε True αν θες καθρέφτισμα, όπως στο collect_data

# ---- Keys (όπως collect_data) ----
VK_A, VK_D, VK_S, VK_ESC = 0x41, 0x44, 0x53, 0x1B
VK_LEFT, VK_RIGHT = 0x25, 0x27
def key_down(vk): return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

# ---- Safe defaults αν δεν υπάρχουν στο config ----
try:
    from config import MIN_YAW_RC
except Exception:
    MIN_YAW_RC = 28
try:
    from config import CUT_FWD_ERR
except Exception:
    CUT_FWD_ERR = 0.55


# ---------- YAW mapping (policy) ----------
def steer_to_yaw(steer: float, forward_speed: int):
    s = float(steer)
    if abs(s) < STEER_DEADBAND:
        s_cmd = 0.0
    else:
        s_cmd = (abs(s) - STEER_DEADBAND) / (1.0 - STEER_DEADBAND)
        s_cmd *= np.sign(s)
    if INVERT_YAW:
        s_cmd = -s_cmd
    fb = 0 if abs(s_cmd) > CUT_FWD_ERR else int(forward_speed)
    yaw = int(np.clip(s_cmd * YAW_GAIN * MAX_YAW_RC, -MAX_YAW_RC, MAX_YAW_RC))
    if yaw != 0 and abs(yaw) < MIN_YAW_RC:
        yaw = MIN_YAW_RC if yaw > 0 else -MIN_YAW_RC
    lr = 0
    ud = 0
    return lr, fb, ud, yaw, s_cmd


# ---------- Quick retrain + snapshots ----------
def quick_retrain(agg_csv: Path, out_models: Path, out_session: Path):
    from torch.utils.data import DataLoader
    from torch import nn
    from torch.optim import Adam
    from dataset import SteeringDataset

    model = SmallCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print("[DAgger] Warning: could not load base MODEL_PATH:", e)

    did_train = False
    try:
        ds = SteeringDataset(agg_csv, augment=True)
        n = len(ds)
        print(f"[DAgger] Aggregated samples: {n}")
        if n >= 50:
            loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
            optim = Adam(model.parameters(), lr=5e-4)
            loss_fn = nn.SmoothL1Loss()
            model.train()
            for epoch in range(3):
                tot, cnt = 0.0, 0
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    optim.zero_grad(); loss.backward(); optim.step()
                    tot += loss.item() * x.size(0); cnt += x.size(0)
                print(f"[DAgger Retrain] epoch {epoch+1}: loss={tot/max(1,cnt):.4f}")
            did_train = True
        else:
            print("[DAgger] Too few samples for retrain (keeping base policy).")
    except Exception as e:
        print("[DAgger] Retrain skipped due to error:", e)

    out_models.mkdir(parents=True, exist_ok=True)
    out_session.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap   = out_models / f"policy_dagger_{stamp}.pt"
   # latest = out_models / "policy_latest.pt"
    sesscp = out_session / f"policy_after_retrain_{stamp}.pt"

    torch.save(model.state_dict(), snap)
   # torch.save(model.state_dict(), latest)
    torch.save(model.state_dict(), sesscp)
    print(f"[DAgger] Saved snapshots:\n  - {snap}\n   - {sesscp}\n  (trained={did_train})")


def main():
    dagger_dir = DATA_ROOT / f"dagger_{time.strftime('%Y%m%d-%H%M%S')}"
    img_dir = dagger_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = dagger_dir / "labels.csv"
    print(f"[DAgger] Session: {dagger_dir}")

    base_csv = DATA_ROOT / "labels.csv"
    if base_csv.exists() and not agg_csv.exists():
        try:
            pd.read_csv(base_csv).to_csv(agg_csv, index=False)
            print("[DAgger] Initialized labels from base CSV.")
        except Exception as e:
            print("[DAgger] Bootstrap skipped:", e)

    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"[DAgger] Loaded policy: {MODEL_PATH}")

    tello, cap = None, None
    last_alt_t, ud_cmd = 0.0, 0

    try:
        tello = safe_tello(connect_only=True)
        time.sleep(0.5)
        tello.send_rc_control(0,0,0,0); time.sleep(0.2)

        if not robust_takeoff(tello, retries=3, pause=1.5):
            raise RuntimeError("Takeoff failed.")
        tello.send_rc_control(0,0,0,0); time.sleep(0.6)

        tello.streamon(); time.sleep(STREAM_WARMUP_SEC)
        cap = open_cv_stream(TELLO_STREAM_URL)
        if not cap or not cap.isOpened():
            try:
                tello.streamoff(); time.sleep(0.5)
                tello.streamon();  time.sleep(STREAM_WARMUP_SEC)
            except: pass
            cap = open_cv_stream(TELLO_STREAM_URL)
            if not cap or not cap.isOpened():
                raise RuntimeError("OpenCV capture failed to open.")

        for _ in range(15):
            _ = read_frame(cap)

        new_file = not agg_csv.exists() or agg_csv.stat().st_size == 0
        f_csv = open(agg_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        if new_file:
            writer.writerow(["filename", "steer"])

        print("[DAgger] Semi-autonomous ON. A/D/←/→ corrections, S=stop forward, ESC=land.")
        idx = 0

        while True:
            frame = read_frame(cap)
            if frame is None:
                tello.send_rc_control(0,0,0,0); time.sleep(0.02); continue

            if USE_FLIP:
                frame = cv2.flip(frame, 0)

            left  = key_down(VK_A) or key_down(VK_LEFT)
            right = key_down(VK_D) or key_down(VK_RIGHT)
            stop_forward = key_down(VK_S)
            if key_down(VK_ESC):
                print("[DAgger] ESC — landing."); break

            im = cv2.resize(frame, (160, 120))
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                steer = float(model(x).cpu().numpy()[0])

            now = time.time()
            if now - last_alt_t > 0.2:
                try:
                    h = tello.get_height()
                    err = ALT_TARGET_CM - h
                    ud_cmd = int(np.clip(ALT_KP * err, -ALT_UD_LIMIT, ALT_UD_LIMIT))
                except Exception:
                    pass
                last_alt_t = now

            if left ^ right:
                yaw = -int(MAX_YAW_RC) if left else int(MAX_YAW_RC)
                fb  = 0 if stop_forward else FORWARD_SPEED
                lr  = 0
                s_cmd = -1.0 if left else 1.0
                expert_label = s_cmd
            else:
                lr, fb, _, yaw, s_cmd = steer_to_yaw(
                    steer,
                    forward_speed = 0 if stop_forward else FORWARD_SPEED
                )
                expert_label = 0.0

            tello.send_rc_control(lr, fb, ud_cmd, yaw)

            fname = f"ep_{idx:06d}.jpg"
            cv2.imwrite(str(img_dir / fname), frame)
            writer.writerow([f"images/{fname}", f"{expert_label:.4f}"])
            idx += 1

            vis = visualize(frame, steer, info=f"DAgger | steer={steer:+.2f} | expert={expert_label:+.2f} | yaw={yaw} | fwd={fb}")
            cv2.putText(vis, f"ALT tgt={ALT_TARGET_CM}cm ud={ud_cmd}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("DAgger (ESC to land)", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        f_csv.close()
        print("[DAgger] Retraining on aggregated data…")
        quick_retrain(agg_csv, MODELS_DIR, dagger_dir)

    finally:
        try:
            if tello:
                tello.send_rc_control(0,0,0,0); time.sleep(0.1)
                #robust_land(tello)
        except:
            pass
        safe_cleanup(tello, cap)


if __name__ == "__main__":
    main()

