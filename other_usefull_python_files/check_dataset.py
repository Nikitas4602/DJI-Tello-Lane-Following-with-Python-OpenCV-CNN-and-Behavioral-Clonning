# check_dataset.py — Γρήγορος έλεγχος αν τα δεδομένα υπάρχουν/διαβάζονται
from pathlib import Path
import pandas as pd

SESS = Path("data/session_001")  # άλλαξέ το αν έχεις άλλο session
csv_path = SESS / "labels.csv"
img_dir  = SESS / "images"

print("[CHECK] CSV exists:", csv_path.exists(), "| size:", csv_path.stat().st_size if csv_path.exists() else 0)
print("[CHECK] Images dir exists:", img_dir.exists())
print("[CHECK] Images count:", len(list(img_dir.glob("*.jpg"))))

if csv_path.exists() and csv_path.stat().st_size > 0:
    try:
        df = pd.read_csv(csv_path)
        print("[CHECK] CSV rows:", len(df))
        print(df.head(5))
    except Exception as e:
        print("[CHECK] CSV read error:", e)
else:
    print("[CHECK] CSV is empty or missing header.")
