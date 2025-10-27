# collect_data.py — auto new session, fixed altitude 1 m, horizontal flip stream
import csv, time, cv2, ctypes
from pathlib import Path
from utils import (
    safe_tello, open_cv_stream, read_frame, visualize,
    rc_from_steer, safe_cleanup, robust_takeoff, robust_land
)
from config import (
    SESSION_DIR, FORWARD_SPEED, YAW_GAIN, MAX_YAW_RC,
    TELLO_STREAM_URL, STREAM_WARMUP_SEC,
    ALT_TARGET_CM, ALT_KP, ALT_UD_LIMIT
)

# Windows Virtual-Key codes
VK_A, VK_D, VK_S, VK_ESC = 0x41, 0x44, 0x53, 0x1B
VK_LEFT, VK_RIGHT = 0x25, 0x27
def key_down(vk): return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

def main():
    session_dir = SESSION_DIR
    img_dir = session_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "labels.csv"

    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if new_file: writer.writerow(["filename","steer"])

    print(f"[INFO] New session: {session_dir.name}")
    print("[INFO] Collect at fixed altitude (100 cm). Controls: A/D turn, S stop forward, ESC to land.")

    tello=None; cap=None
    steer=0.0; idx=0
    last_alt_t=0.0; ud_cmd=0
    alt_target_cm = 100   # σταθερό 1 m

    try:
        tello = safe_tello(connect_only=True)
        time.sleep(0.5)
        tello.send_rc_control(0,0,0,0); time.sleep(0.2)

        if not robust_takeoff(tello, retries=3, pause=1.5):
            raise RuntimeError("Απέτυχε η απογείωση.")

        # bring to ~1m (προσεκτική προσέγγιση προς τα κάτω αν ήδη ψηλά)
        try:
            h0 = tello.get_height()
            delta = max(0, h0 - alt_target_cm)
            if delta >= 20:
                tello.move_down(int(delta)); time.sleep(0.8)
        except Exception: pass

        tello.send_rc_control(0,0,0,0); time.sleep(0.6)

        # video stream
        tello.streamon(); time.sleep(STREAM_WARMUP_SEC)
        cap = open_cv_stream(TELLO_STREAM_URL)
        if not cap or not cap.isOpened():
            try:
                tello.streamoff(); time.sleep(0.5)
                tello.streamon(); time.sleep(STREAM_WARMUP_SEC)
            except: pass
            cap = open_cv_stream(TELLO_STREAM_URL)
            if not cap or not cap.isOpened():
                raise RuntimeError("OpenCV capture δεν άνοιξε (udp 11111).")

        # μικρό warmup
        t0=time.time()
        while time.time()-t0 < 1.0: _=read_frame(cap)

        while True:
            frame = read_frame(cap)
            if frame is None:
                tello.send_rc_control(0,0,0,0); time.sleep(0.02); continue

            # ✅ οριζόντιο flip (δεξιά⇄αριστερά) στο stream
            frame = cv2.flip(frame, 1)

            # --- ΠΛΗΚΤΡΑ ---
            left  = key_down(VK_A) or key_down(VK_LEFT)
            right = key_down(VK_D) or key_down(VK_RIGHT)
            stop_forward = key_down(VK_S)

            if key_down(VK_ESC):
                print("[INFO] ESC — landing & save."); break

            # --- LABEL ΓΙΑ CSV (κρατάμε ορολογία 'steer') ---
            # A → -1, D → +1, τίποτα → 0 (με μικρή εξομάλυνση για σταθερό label)
            target = (-1.0 if left else 0.0) + (1.0 if right else 0.0)
            steer = 0.7*steer + 0.3*target

            # --- ALT HOLD γύρω από 100 cm ---
            now=time.time()
            if now-last_alt_t>0.2:
                try:
                    h = tello.get_height()
                    err = alt_target_cm - h
                    ud_cmd = int(max(-ALT_UD_LIMIT, min(ALT_UD_LIMIT, ALT_KP * err)))
                except Exception: pass
                last_alt_t = now

            # --- ΕΝΤΟΛΕΣ RC ---
            # S = κόβει μπροστά ταχύτητα
            fb = 0 if stop_forward else FORWARD_SPEED
            lr = 0  # δεν χρησιμοποιούμε roll για στρίψιμο
            # A/D = ΑΜΕΣΟΣ έλεγχος YAW (στροφή) χωρίς rc_from_steer
            if left and not right:
                yaw = -int(MAX_YAW_RC)        # αριστερά
            elif right and not left:
                yaw =  int(MAX_YAW_RC)        # δεξιά
            else:
                yaw = 0                        # ίσια

            tello.send_rc_control(lr, fb, ud_cmd, yaw)

            # --- ΑΠΟΘΗΚΕΥΣΗ ΔΕΙΓΜΑΤΟΣ ---
            fname = f"img_{idx:06d}.jpg"
            (img_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_dir/fname), frame)
            writer.writerow([f"images/{fname}", f"{steer:.4f}"])
            idx += 1

            # --- OVERLAY ---
            vis = visualize(frame, steer,
                            info=f"Collect | steer={steer:+.2f} | yaw={yaw} | fwd={fb} | ALT tgt={alt_target_cm}cm ud={ud_cmd}")
            cv2.imshow(f"Collect — {session_dir.name}", vis)
            cv2.waitKey(1)

    except Exception as e:
        print("[ERROR] collect_data:", e)

    finally:
        try:
            if tello:
                tello.send_rc_control(0,0,0,0); time.sleep(0.1)
                robust_land(tello)
        except: pass
        try: f_csv.close()
        except: pass
        safe_cleanup(tello, cap)
        print(f"[INFO] Saved session at: {session_dir}")

if __name__ == "__main__":
    main()
