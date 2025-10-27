# clean_auto_final.py — βρίσκει μόνο του το πιο πρόσφατο session, καθαρίζει και φτιάχνει αυτόματα _final folder
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd

DATA_ROOT = Path("data")

# -------- ρυθμίσεις φίλτρων --------
BOTTOM_RATIO = 0.25
TOP_RATIO = 0.60
CANNY_LOW, CANNY_HIGH = 60, 140
SMOOTH_WIN = 21
PEAK_FRAC = 0.18
PEAK_MIN_DIST_FRAC = 0.22
TOP_HOUGH_THRESHOLD = 55
HOUGH_MIN_LEN = 35
HOUGH_MAX_GAP = 12
SAVE_REJECT_SAMPLES = True
REJECT_SAMPLES_MAX = 40
# -----------------------------------

def find_latest_session():
    if not DATA_ROOT.exists():
        print("[ERROR] Δεν υπάρχει φάκελος data/")
        return None
    sessions = sorted(
        [p for p in DATA_ROOT.glob("session_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        print("[ERROR] Δεν βρέθηκε κανένα session.")
        return None
    latest = sessions[0]
    print(f"[INFO] Τελευταίο session: {latest.name}")
    return latest

def make_unique_dir(base: Path) -> Path:
    """Φτιάχνει μοναδικό φάκελο: base, base2, base3, ..."""
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
    yb = int(h * (1.0 - BOTTOM_RATIO))
    roi_bot = img[yb:, :]
    edges_b = cv2.Canny(cv2.cvtColor(roi_bot, cv2.COLOR_BGR2GRAY), CANNY_LOW, CANNY_HIGH)
    if not hist_peaks_two(edges_b):
        return False
    yt = int(h * TOP_RATIO)
    roi_top = img[:yt, :]
    edges_t = cv2.Canny(cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY), CANNY_LOW, CANNY_HIGH)
    lines_t = cv2.HoughLinesP(
        edges_t, 1, np.pi / 180, threshold=60,
        minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP,
    )
    if lines_t is not None and len(lines_t) > TOP_HOUGH_THRESHOLD:
        return False
    return True

def main():
    src_session = find_latest_session()
    if src_session is None:
        return

    src_csv = src_session / "labels.csv"
    src_img_dir = src_session / "images"
    if not src_csv.exists() or not src_img_dir.exists():
        print("[ERROR] Δεν βρέθηκαν labels.csv ή images/ στο:", src_session)
        return

    dst_session = make_unique_dir(Path(str(src_session) + "_final"))
    dst_img_dir = dst_session / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_session / "labels.csv"

    if SAVE_REJECT_SAMPLES:
        rej_dir = dst_session / "rejected_samples"
        rej_dir.mkdir(parents=True, exist_ok=True)
        saved_rej = 0
    else:
        rej_dir = None
        saved_rej = 0

    df = pd.read_csv(src_csv)
    print(f"[INFO] Πηγή: {src_session}")
    print(f"[INFO] Τελικός φάκελος: {dst_session}")
    print(f"[INFO] Γραμμές CSV: {len(df)}")

    kept_rows, kept, rejected = [], 0, 0

    for i, row in df.iterrows():
        rel = str(row["filename"])
        img_path = src_session / rel
        if not img_path.exists():
            rejected += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            rejected += 1
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
    print(f"  Νέο set:  {dst_session}")

if __name__ == "__main__":
    main()
