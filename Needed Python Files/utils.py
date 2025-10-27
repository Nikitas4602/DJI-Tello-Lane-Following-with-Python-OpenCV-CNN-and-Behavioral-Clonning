import time
import cv2
from djitellopy import Tello

def safe_tello(connect_only=False):
    """Σύνδεση, set_speed. Αν connect_only=True, ΔΕΝ ανοίγει stream εδώ."""
    t = Tello()
    t.connect()
    t.set_speed(20)
    if connect_only:
        return t
    return t

def open_cv_stream(url: str):
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

def read_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None

def robust_takeoff(t: Tello, retries=3, pause=1.5):
    try:
        bat = t.get_battery()
        print(f"[INFO] Battery: {bat}%")
        if bat is not None and bat < 15:
            raise RuntimeError("Χαμηλή μπαταρία (<15%). Φόρτισέ το.")
    except Exception as e:
        print("[WARN] Battery read failed:", e)
    for i in range(1, retries+1):
        try:
            print(f"[INFO] takeoff try {i}/{retries}")
            t.takeoff()
            time.sleep(pause)
            return True
        except Exception as e:
            print(f"[WARN] takeoff failed (try {i}): {e}")
            time.sleep(pause)
    return False

def robust_land(t: Tello, retries=2):
    for i in range(1, retries+1):
        try:
            t.land()
            return True
        except Exception as e:
            print(f"[WARN] land failed (try {i}): {e}")
            time.sleep(0.8)
    return False

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rc_from_steer(steer, forward=20, yaw_gain=80, yaw_limit=70):
    yaw = int(clamp(steer * yaw_gain, -yaw_limit, yaw_limit))
    return 0, int(forward), 0, yaw  # (lr, fb, ud, yaw)

def visualize(frame, steer, info=""):
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    cx, cy = w//2, h-20
    dx = int(steer * w * 0.4)
    cv2.arrowedLine(overlay, (cx, cy), (cx+dx, cy-60), (0,255,0), 3)
    if info:
        cv2.putText(overlay, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return overlay

def safe_cleanup(tello: Tello = None, cap=None):
    try:
        if tello:
            tello.send_rc_control(0,0,0,0)
            time.sleep(0.1)
    except: pass
    try:
        if cap: cap.release()
    except: pass
    try:
        if tello: tello.streamoff()
    except: pass
    try:
        cv2.destroyAllWindows()
    except: pass
    try:
        if tello: tello.end()
    except: pass
