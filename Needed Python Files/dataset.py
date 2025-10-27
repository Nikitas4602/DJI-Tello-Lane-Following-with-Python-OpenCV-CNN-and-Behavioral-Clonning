from pathlib import Path
import cv2, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset
#from config import IMG_W, IMG_H, ROI_CROP_TOP_RATIO
from config import IMG_W, IMG_H

class SteeringDataset(Dataset):
    def __init__(self, csv_path, root=None, augment=False):
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root else self.csv_path.parent
        self.augment = augment

        if not self.csv_path.exists():
            print(f"[ERROR] CSV not found: {self.csv_path}")
            self.samples = []; return
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {self.csv_path} -> {e}")
            self.samples = []; return
        if "filename" not in df.columns or "steer" not in df.columns:
            print(f"[ERROR] CSV missing columns (filename, steer): {self.csv_path}")
            self.samples = []; return

        total = len(df); self.samples = []; missing = 0
        for _, row in df.iterrows():
            rel = str(row["filename"])
            img_path = self.root / rel
            if not img_path.exists():
                missing += 1; continue
            steer = float(row["steer"])
            self.samples.append((img_path, steer))
        print(f"[DATASET] {self.csv_path} | total: {total}, usable: {len(self.samples)}, missing: {missing}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, steer = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        # ROI crop στο κάτω μέρος
        if 0.0 <= ROI_CROP_TOP_RATIO < 0.9:
            h = img.shape[0]
            top = int(h * ROI_CROP_TOP_RATIO)
            img = img[top:, :]
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        x = torch.from_numpy(img.transpose(2,0,1))
        y = torch.tensor([steer], dtype=torch.float32)
        return x, y
