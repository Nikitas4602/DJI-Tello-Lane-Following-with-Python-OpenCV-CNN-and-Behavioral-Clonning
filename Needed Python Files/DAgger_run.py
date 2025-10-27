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
    # policy & device
    MODEL_PATH, DEVICE,
    # stream
    TELLO_STREAM_URL, STREAM_WARMUP_SEC,
    # control gainsaaaaaaaaddddddddddddaa
    INVERT_YAW, STEER_DEADBAND, YAW_GAIN, MAX_YAW_RC,
    FORWARD_SPEED,
    # altitude hold
    ALT_TARGET_CM, ALT_KP, ALT_UD_LIMIT,
)

# ---- Optional flip (OFF by default) ----
USE_FLIP = False  # βάλε True αν θες καθρέφτισμα, όπως στο collect_data

# ---- Keys (όπως collect_data) ----
VK_A, VK_D, VK_S, VK_ESC = 0x41, 0x44, 0x53, 0x1B
VK_LEFT, VK_RIGHT = 0x25, 0x27
def key_down(vk): return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

# ---- Safe defaults αν δεν υπάρχουν στο config ----
try:
    from config import MIN_YAW_RC
except Exception:
    MIN_YAW_RC = 28       # ελάχιστο RC για να «πάρει» στροφή
try:
    from config import CUT_FWD_ERR
except Exception:
    CUT_FWD_ERR = 0.55    # κόψε εμπρός όταν το σφάλμα είναι μεγάλο

# ---- DAgger session ρίζα (γράφει εδώ μέσα) ----
DATA_ROOT = Path("data")


# ---------- YAW mapping (policy) ----------
def steer_to_yaw(steer: float, forward_speed: int):
    """
    Μετατρέπει continuous steer [-1..1] σε καθαρό YAW RC (όχι strafe),
    με deadband, min turn και cut-forward όταν το λάθος είναι μεγάλο.
    """
    s = float(steer)

    # deadband + re-scale στο [-1, 1]
    if abs(s) < STEER_DEADBAND:
        s_cmd = 0.0
    else:
        s_cmd = (abs(s) - STEER_DEADBAND) / (1.0 - STEER_DEADBAND)
        s_cmd *= np.sign(s)

    # αντίστροφο αν χρειάζεται
    if INVERT_YAW:
        s_cmd = -s_cmd

    # κόψε την προώθηση αν το σφάλμα είναι μεγάλο
    fb = 0 if abs(s_cmd) > CUT_FWD_ERR else int(forward_speed)

    # χαρτογράφηση σε yaw
    yaw = int(np.clip(s_cmd * YAW_GAIN * MAX_YAW_RC, -MAX_YAW_RC, MAX_YAW_RC))

    # ελάχιστη ώθηση για να ξεκινήσει η στροφή
    if yaw != 0 and abs(yaw) < MIN_YAW_RC:
        yaw = MIN_YAW_RC if yaw > 0 else -MIN_YAW_RC

    lr = 0
    ud = 0
    return lr, fb, ud, yaw, s_cmd


# ---------- Quick retrain + snapshots ----------
def quick_retrain(agg_csv: Path, out_models: Path, out_session: Path):
    """
    Κάνει ένα γρήγορο fine-tune στο DAgger dataset και ΑΠΟΘΗΚΕΥΕΙ:
      - models/policy_dagger_<timestamp>.pt (κρατάς ιστορικό)
      - models/policy_latest.pt (εύκολο για inference)
      - <session>/policy_after_retrain_<timestamp>.pt (session copy)
    Δεν διαγράφει KΑΝΕΝΑ παλιό policy.
    """
    from torch.utils.data import DataLoader
    from torch import nn
    from torch.optim import Adam
    from dataset import SteeringDataset

    try:
        ds = SteeringDataset(agg_csv, augment=True)
    except Exception as e:
        print("[DAgger] Could not load dataset for retrain:", e)
        return

    if len(ds) < 200:
        print("[DAgger] Not enough samples to retrain.")
        return

    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    model = SmallCNN().to(DEVICE)
    try:
        if Path(MODEL_PATH).exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print("[DAgger] Warning: failed to load base policy:", e)

    optim = Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(3):
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        print(f"[DAgger Retrain] epoch {epoch+1}: loss={tot/max(1,n):.4f}")

    out_models.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap = out_models / f"policy_dagger_{stamp}.pt"
    latest = out_models / "policy_latest.pt"

    torch.save(model.state_dict(), snap)
    torch.save(model.state_dict(), latest)
    print(f"[DAgger] Saved:\n  - {snap}\n  - {latest}")

    out_session.mkdir(parents=True, exist_ok=True)
    session_copy = out_session / f"policy_after_retrain_{stamp}.pt"
    torch.save(model.state_dict(), session_copy)
    print(f"[DAgger] Session copy: {session_copy}")


def main():
    # ----- DAgger session paths -----
    dagger_dir = DATA_ROOT / f"dagger_{time.strftime('%Y%m%d-%H%M%S')}"
    img_dir = dagger_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = dagger_dir / "labels.csv"
    models_dir = Path("models")
    print(f"[DAgger] Session: {dagger_dir}")

    # Bootstrap: αν υπάρχει BC CSV δίπλα (προαιρετικό)
    base_csv = DATA_ROOT / "labels.csv"
    if base_csv.exists() and not agg_csv.exists():
        try:
            pd.read_csv(base_csv).to_csv(agg_csv, index=False)
            print("[DAgger] Initialized labels from base CSV.")
        except Exception as e:
            print("[DAgger] Bootstrap skipped:", e)

    # ----- Load policy (από MODEL_PATH) -----
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"[DAgger] Loaded policy: {MODEL_PATH}")

    tello, cap = None, None
    last_alt_t, ud_cmd = 0.0, 0

    try:
        # ----- Connect & take off -----
        tello = safe_tello(connect_only=True)
        time.sleep(0.5)
        tello.send_rc_control(0,0,0,0); time.sleep(0.2)

        if not robust_takeoff(tello, retries=3, pause=1.5):
            raise RuntimeError("Takeoff failed.")
        tello.send_rc_control(0,0,0,0); time.sleep(0.6)

        # ----- Stream startup like inference -----
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

        # warmup frames
        for _ in range(15):
            _ = read_frame(cap)

        # CSV writer (append-safe)
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

            # optional flip (OFF by default)
            if USE_FLIP:
                frame = cv2.flip(frame, 1)

            # ---- Expert keys (όπως collect_data) ----
            left  = key_down(VK_A) or key_down(VK_LEFT)
            right = key_down(VK_D) or key_down(VK_RIGHT)
            stop_forward = key_down(VK_S)
            if key_down(VK_ESC):
                print("[DAgger] ESC — landing."); break

            # ---- Policy steer ----
            im = cv2.resize(frame, (160, 120))
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                steer = float(model(x).cpu().numpy()[0])

            # ---- Alt-hold ----
            now = time.time()
            if now - last_alt_t > 0.2:
                try:
                    h = tello.get_height()
                    err = ALT_TARGET_CM - h
                    ud_cmd = int(np.clip(ALT_KP * err, -ALT_UD_LIMIT, ALT_UD_LIMIT))
                except Exception:
                    pass
                last_alt_t = now

            # ---- RC commands ----
            # Αν πατιούνται A/D → ΑΜΕΣΟΣ έλεγχος yaw (expert), αλλιώς policy με YAW mapping
            if left ^ right:
                yaw = -int(MAX_YAW_RC) if left else int(MAX_YAW_RC)
                fb  = 0 if stop_forward else FORWARD_SPEED
                lr  = 0
                s_cmd = -1.0 if left else 1.0
                expert_label = s_cmd  # γράφουμε στο CSV το expert (-1, 0, +1)
            else:
                lr, fb, _, yaw, s_cmd = steer_to_yaw(
                    steer,
                    forward_speed = 0 if stop_forward else FORWARD_SPEED
                )
                expert_label = 0.0  # όταν δεν διορθώνεις, expert=0 (καμία παρέμβαση)

            tello.send_rc_control(lr, fb, ud_cmd, yaw)

            # ---- Save sample (frame + expert label) ----
            fname = f"ep_{idx:06d}.jpg"
            cv2.imwrite(str(img_dir / fname), frame)
            writer.writerow([f"images/{fname}", f"{expert_label:.4f}"])
            idx += 1

            # ---- Overlay ----
            vis = visualize(frame, steer, info=f"DAgger | steer={steer:+.2f} | expert={expert_label:+.2f} | yaw={yaw} | fwd={fb}")
            cv2.putText(vis, f"ALT tgt={ALT_TARGET_CM}cm ud={ud_cmd}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("DAgger (ESC to land)", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        f_csv.close()

        # ---- Quick retrain + snapshots (ΔΕΝ διαγράφει τίποτα) ----
        print("[DAgger] Retraining on aggregated data…")
        quick_retrain(agg_csv, models_dir, dagger_dir)

    finally:
        try:
            if tello:
                tello.send_rc_control(0,0,0,0); time.sleep(0.1)
                robust_land(tello)
        except:
            pass
        safe_cleanup(tello, cap)


if __name__ == "__main__":
    main()
