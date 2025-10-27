import time, cv2
from djitellopy import Tello

me = Tello()
me.connect()
print("Battery:", me.get_battery())
me.streamon()
time.sleep(2)
fr = me.get_frame_read()

for i in range(200):
    frame = fr.frame
    if frame is None or frame.size == 0:
        time.sleep(0.01)
        continue
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

me.streamoff()
cv2.destroyAllWindows()
