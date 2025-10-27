# crop_bottom_half.py
# Κόβει όλες τις εικόνες ενός session και κρατά ΜΟΝΟ το κάτω μισό
# Δημιουργεί αυτόματα νέο φάκελο <session>_cropped

from pathlib import Path
import cv2
import shutil
import pandas as pd

# === ΒΑΛΕ ΕΔΩ ΤΟ SESSION ΠΟΥ ΘΕΣ ΝΑ ΕΠΕΞΕΡΓΑΣΤΕΙ ===
SOURCE_SESSION = Path(r"data\session_2025-10-14_13-54-28_clean_final")
# =====================================================

def make_unique_dir(base: Path) -> Path:
    """Αν υπάρχει ήδη φάκελος, φτιάχνει base2, base3 κλπ."""
    if not base.exists():
        return base
    i = 2
    while True:
        cand = Path(str(base) + str(i))
        if not cand.exists():
            return cand
        i += 1

def main():
    src_csv = SOURCE_SESSION / "labels.csv"
    src_img_dir = SOURCE_SESSION / "images"

    if not src_csv.exists() or not src_img_dir.exists():
        print("[ERROR] Δεν βρέθηκαν labels.csv ή images/ στο:", SOURCE_SESSION)
        return

    dst_session = make_unique_dir(Path(str(SOURCE_SESSION) + "_cropped"))
    dst_img_dir = dst_session / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_session / "labels.csv"

    df = pd.read_csv(src_csv)
    kept_rows = []
    kept, failed = 0, 0

    print(f"[INFO] Πηγή: {SOURCE_SESSION}")
    print(f"[INFO] Νέος φάκελος: {dst_session}")
    print(f"[INFO] Εικόνες προς επεξεργασία: {len(df)}")

    for i, row in df.iterrows():
        rel = str(row["filename"])
        img_path = SOURCE_SESSION / rel
        if not img_path.exists():
            failed += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue

        # --- Κόβει το κάτω μισό ---
        h = img.shape[0]
        bottom_half = img[h // 2 :, :]

        # --- Αποθηκεύει τη νέα εικόνα ---
        dst_path = dst_img_dir / img_path.name
        cv2.imwrite(str(dst_path), bottom_half)

        kept_rows.append([f"images/{img_path.name}", row["steer"]])
        kept += 1

        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(df)} | saved={kept}")

    # --- Νέο CSV ---
    pd.DataFrame(kept_rows, columns=["filename", "steer"]).to_csv(dst_csv, index=False)

    print("\n[DONE]")
    print(f"  Saved cropped images: {kept}")
    print(f"  Failed: {failed}")
    print(f"  Νέος φάκελος: {dst_session}")

if __name__ == "__main__":
    main()
