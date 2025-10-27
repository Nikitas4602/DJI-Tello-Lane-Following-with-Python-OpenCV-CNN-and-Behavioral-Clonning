# clean_simple.py
# Απλό script: καθαρίζει ένα συγκεκριμένο session και φτιάχνει μόνο του νέο φάκελο "_final"





#####ΠΟΛΥ ΚΑΚΟ########






from pathlib import Path
import shutil, cv2, numpy as np, pandas as pd

# ==========================================================
# ΕΔΩ ΒΑΖΕΙΣ ΤΟ SESSION ΠΟΥ ΘΕΣ ΝΑ ΚΑΘΑΡΙΣΤΕΙ
SOURCE_SESSION = Path("data/session_2025-10-14_13-54-28")
# ==========================================================

BOTTOM_RATIO = 0.25
TOP_RATIO = 0.60
CANNY_LOW, CANNY_HIGH = 60, 140
SMOOTH_WIN = 21
PEAK_FRAC = 0.18
PEAK_MIN_DIST_FRAC = 0.22
TOP_HOUGH_THRESHOLD = 55
HOUGH_MIN_LEN = 35
HOUGH_MAX_GAP = 12
MID_BAND = (0.40, 0.70)
MID_MIN_LINE_LEN = 60
MID_MAX_GAP = 10
HORIZ_ANGLE_DEG = 15
SAVE_REJECT_SAMPLES = True
REJECT_SAMPLES_MAX = 40


def make_unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 2
    while True:
        cand = Path(str(base) + str(i))
        if not cand.exists():
            return cand
        i += 1


def hist_peaks_two(edges):
    col_hist = edges.sum(axis=0).astype(np.float32)
    if col_hist.max() < 1:
        return False
    k = SMOOTH_WIN if SMOOTH_WIN % 2 == 1 else SMOOTH_WIN + 1
    col_hist_s = cv2.GaussianBlur(col_hist.reshape(1, -1), (k, 1), 0).ravel()
    thr = PEAK_FRAC * col_hist_s.max()
    peaks = []
    for i in range(1, len(col_hist_s) - 1):
        if col_hist_s[i] > thr and col_hist_s[i] >= col_hist_s[i - 1] and col_hist_s[i] >= col_hist_s[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return False
    peaks_sorted = sorted(peaks, key=lambda i: col_hist_s[i], reverse=True)
    p1, p2 = peaks_sorted[0], peaks_sorted[1]
    min_dist = int(PEAK_MIN_DIST_FRAC * edges.shape[1])
    return abs(p1 - p2) >= min_dist


def want_keep(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- κάτω μέρος: πρέπει να έχει 2 γραμμές ---
    yb = int(h * (1.0 - BOTTOM_RATIO))
    roi_bot = gray[yb:, :]
    edges_b = cv2.Canny(roi_bot, CANNY_LOW, CANNY_HIGH)
    if not hist_peaks_two(edges_b):
        return False

    # --- μεσαία ζώνη: απορρίπτουμε οριζόντιες μπάρες ---
    y1, y2 = int(h * MID_BAND[0]), int(h * MID_BAND[1])
    mid = gray[y1:y2, :]
    edges_m = cv2.Canny(mid, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges_m, 1, np.pi / 180, threshold=50,
                            minLineLength=MID_MIN_LINE_LEN, maxLineGap=MID_MAX_GAP)
    if lines is not None:
        for (x1, y1m, x2, y2m) in lines[:, 0, :]:
            angle = abs(np.degrees(np.arctan2(y2m - y1m, x2 - x1)))
            if angle <= HORIZ_ANGLE_DEG:
                return False

    # --- πάνω μέρος: απορρίπτουμε χαμό ---
    yt = int(h * TOP_RATIO)
    roi_top = gray[:yt, :]
    edges_t = cv2.Canny(roi_top, CANNY_LOW, CANNY_HIGH)
    lines_t = cv2.HoughLinesP(edges_t, 1, np.pi / 180, threshold=60,
                              minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    if lines_t is not None and len(lines_t) > TOP_HOUGH_THRESHOLD:
        return False

    return True


def main():
    src_csv = SOURCE_SESSION / "labels.csv"
    src_img_dir = SOURCE_SESSION / "images"

    if not src_csv.exists() or not src_img_dir.exists():
        print("[ERROR] Δεν βρέθηκαν labels.csv ή images/ στο:", SOURCE_SESSION)
        return

    dst_session = make_unique_dir(Path(str(SOURCE_SESSION) + "_final"))
    dst_img_dir = dst_session / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_session / "labels.csv"

    if SAVE_REJECT_SAMPLES:
        rej_dir = dst_session / "rejected_samples"
        rej_dir.mkdir(parents=True, exist_ok=True)
    else:
        rej_dir = None

    df = pd.read_csv(src_csv)
    kept_rows, kept, rejected, saved_rej = [], 0, 0, 0

    print(f"[INFO] Καθαρίζω session: {SOURCE_SESSION.name}")
    print(f"[INFO] Γραμμές CSV: {len(df)}")

    for i, row in df.iterrows():
        rel = str(row["filename"])
        img_path = SOURCE_SESSION / rel
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if want_keep(img):
            dst_path = dst_img_dir / img_path.name
            shutil.copy2(img_path, dst_path)
            kept_rows.append([f"images/{img_path.name}", row["steer"]])
            kept += 1
        else:
            rejected += 1
            if rej_dir is not None and saved_rej < REJECT_SAMPLES_MAX:
                cv2.imwrite(str(rej_dir / img_path.name), img)
                saved_rej += 1

        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(df)} | kept={kept} | rejected={rejected}")

    pd.DataFrame(kept_rows, columns=["filename", "steer"]).to_csv(dst_csv, index=False)
    print("\n[DONE]")
    print(f"  Kept:     {kept}")
    print(f"  Rejected: {rejected}")
    print(f"  Νέο session: {dst_session}")


if __name__ == "__main__":
    main()
#