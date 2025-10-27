import cv2
import numpy as np

def steer_between_lines(frame, debug=False):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges[:h//2, :] = 0  # Κρατάμε μόνο το κάτω μισό

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=60, maxLineGap=20)

    left_x, right_x = [], []
    y_ref = int(h * 0.9)

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 10:
                x_at = x1
            else:
                m = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(m) < 0.5:
                    continue
                b = y1 - m * x1
                x_at = int((y_ref - b) / m)

            if 0 <= x_at < w:
                (left_x if x_at < w // 2 else right_x).append(x_at)

    if len(left_x) == 0 or len(right_x) == 0:
        return None, frame if debug else None

    xL = int(np.median(left_x))
    xR = int(np.median(right_x))
    mid = (xL + xR) // 2
    cx = w // 2
    err = (mid - cx) / (w * 0.5)
    steer = float(np.clip(err, -1, 1))

    if debug:
        vis = frame.copy()
        cv2.line(vis, (xL, y_ref - 40), (xL, y_ref + 40), (0, 255, 255), 2)
        cv2.line(vis, (xR, y_ref - 40), (xR, y_ref + 40), (0, 255, 255), 2)
        cv2.arrowedLine(vis, (cx, y_ref), (mid, y_ref), (0, 255, 0), 3)
        return steer, vis

    return steer, None
