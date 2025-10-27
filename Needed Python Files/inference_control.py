# inference_control.py — proper yaw control for autonomous turning
import time, cv2, torch, ctypes, numpy as np
from model import SmallCNN
from utils import (safe_tello, open_cv_stream, read_frame, visualize,
                   safe_cleanup, robust_takeoff, robust_land)
from config import (
    MODEL_PATH, DEVICE,
    TELLO_STREAM_URL, STREAM_WARMUP_SEC,
    ALT_TARGET_CM, ALT_KP, ALT_UD_LIMIT,
    INVERT_YAW, STEER_DEADBAND, YAW_GAIN, MAX_YAW_RC, MIN_YAW_RC,
    FORWARD_SPEED, CUT_FWD_ERR
)
#Eπέλεξε το μοντέλο που θες να χρησιμοποιήσεις
MODEL_PATH = "models\policy_2025-10-22_11-51-51_good_best"

# keyboard για altitude αλλαγές
VK_UP, VK_DOWN = 0x26, 0x28
def key_down(vk): return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

def steer_to_yaw(steer: float):
    """Μετατρέπει continuous steer [-1..1] σε yaw RC εντολή"""
    s = float(steer)

    # Νεκρή ζώνη
    if abs(s) < STEER_DEADBAND:
        s_cmd = 0.0
    else:
        s_cmd = (abs(s) - STEER_DEADBAND) / (1.0 - STEER_DEADBAND)
        s_cmd *= np.sign(s)

    # Αντίστροφο αν χρειάζεται
    if INVERT_YAW:
        s_cmd = -s_cmd

    # Προώθηση — κόψε όταν υπάρχει μεγάλο σφάλμα
    fb = 0 if abs(s_cmd) > CUT_FWD_ERR else int(FORWARD_SPEED)

    # Yaw RC mapping
    yaw = int(np.clip(s_cmd * YAW_GAIN * MAX_YAW_RC, -MAX_YAW_RC, MAX_YAW_RC))

    # Ελάχιστη δύναμη για να «ξεκινάει» στροφή
    if yaw != 0 and abs(yaw) < MIN_YAW_RC:
        yaw = MIN_YAW_RC if yaw > 0 else -MIN_YAW_RC

    # Τίποτα άλλο δεν αλλάζουμε
    lr = 0
    ud = 0
    return lr, fb, ud, yaw, s_cmd

def main():
    # Φόρτωση μοντέλου
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    tello, cap = None, None
    last_alt_t, ud_cmd = 0.0, 0
    alt_target_cm = ALT_TARGET_CM

    try:
        tello = safe_tello(connect_only=True)
        time.sleep(0.5)
        tello.send_rc_control(0,0,0,0); time.sleep(0.2)

        if not robust_takeoff(tello, retries=3, pause=1.5):
            raise RuntimeError("Απέτυχε η απογείωση.")
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
                raise RuntimeError("OpenCV capture δεν άνοιξε.")

        # Ζέσταμα stream
        t0 = time.time()
        while time.time() - t0 < 1.0:
            _ = read_frame(cap)

        print("[RUN] Autonomous (YAW TURN). ESC για προσγείωση. ALT: ↑/↓")

        while True:
            frame = read_frame(cap)
            if frame is None:
                tello.send_rc_control(0,0,0,0); time.sleep(0.02)
                continue

            # ALTITUDE adjustment
            if key_down(VK_UP):   alt_target_cm = min(80, alt_target_cm + 5)
            if key_down(VK_DOWN): alt_target_cm = max(15, alt_target_cm - 5)

            # CNN predict steer
            im = cv2.resize(frame, (160,120))
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32")/255.0
            x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                steer = float(model(x).cpu().numpy()[0])

            # Alt-hold
            now = time.time()
            if now - last_alt_t > 0.2:
                try:
                    h = tello.get_height()
                    err = alt_target_cm - h
                    ud_cmd = int(np.clip(ALT_KP * err, -ALT_UD_LIMIT, ALT_UD_LIMIT))
                except Exception:
                    pass
                last_alt_t = now

            # Steering → yaw RC
            lr, fb, _, yaw, s_cmd = steer_to_yaw(steer)
            tello.send_rc_control(lr, fb, ud_cmd, yaw)

            # Overlay
            vis = visualize(frame, steer, info=f"steer={steer:+.2f} | s_cmd={s_cmd:+.2f} | yaw={yaw} | fb={fb}")
            cv2.putText(vis, f"ALT tgt={alt_target_cm}cm ud={ud_cmd}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Autonomous (YAW TURN) - ESC to land", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

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
