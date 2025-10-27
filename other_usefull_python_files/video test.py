# video_test.py (σταθερή εκδοχή)
import time, cv2
from djitellopy import Tello

t = Tello()
try:
    t.connect()
    print("Battery:", t.get_battery())
    t.streamon()
    time.sleep(2)  # δώσε χρόνο να ανοίξει το stream

    # OpenCV capture στο UDP (χωρίς '@')
    cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV capture not opened (11111)")

    print("Press ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue

        cv2.imshow("Tello Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("ERROR:", e)

finally:
    # Ασφαλές κλείσιμο: πάντα μέσα σε try για να μην ξαναπετάξει WinError 6
    try:
        if 'cap' in locals(): cap.release()
    except: pass
    try:
        t.send_rc_control(0,0,0,0)
    except: pass
    try:
        t.streamoff()
    except: pass
    try:
        cv2.destroyAllWindows()
    except: pass
    try:
        t.end()  # κλείνει sockets
    except: pass
