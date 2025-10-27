import pandas as pd
from pathlib import Path
import shutil
import cv2
import numpy as np

# === ΕΠΕΞΕΡΓΑΣΙΑ ΠΑΛΙΟΥ SESSION ===
SOURCE_SESSION = Path("data\session_2025-10-14_13-54-28")  # βάλε το session που θες να καθαρίσεις
DEST_SESSION = Path(str(SOURCE_SESSION) + "_clean")

SRC_CSV = SOURCE_SESSION / "labels.csv"
SRC_IMG_DIR = SOURCE_SESSION / "images"

DST_CSV = DEST_SESSION / "labels.csv"
DST_IMG_DIR = DEST_SESSION / "images"
DST_IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Διαβάζω CSV: {SRC_CSV}")
df = pd.read_csv(SRC_CSV)
print(f"[INFO] Συνολικά δείγματα: {len(df)}")

valid_rows = []
removed = 0

for i, row in df.iterrows():
    src_path = SOURCE_SESSION / row["filename"]
    if not src_path.exists():
        removed += 1
        continue

    # --- Προαιρετικό φίλτρο για “μπερδεμένες” εικόνες ---
    # Παράδειγμα: αν βρεθούν >80 γραμμές με Hough, τη θεωρεί "θορυβώδη"
    img = cv2.imread(str(src_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=15)

    if lines is not None and len(lines) > 80:
        removed += 1
        continue

    # Αν περάσει το φίλτρο → κράτησέ την
    dst_rel = f"images/{src_path.name}"
    dst_path = DST_IMG_DIR / src_path.name
    shutil.copy2(src_path, dst_path)
    valid_rows.append([dst_rel, row["steer"]])

# --- Αποθήκευση νέου CSV ---
clean_df = pd.DataFrame(valid_rows, columns=["filename", "steer"])
clean_df.to_csv(DST_CSV, index=False)

print(f"[DONE] Καθαρές εικόνες: {len(clean_df)} | Αφαιρέθηκαν: {removed}")
print(f"[INFO] Νέο dataset: {DEST_SESSION}")
