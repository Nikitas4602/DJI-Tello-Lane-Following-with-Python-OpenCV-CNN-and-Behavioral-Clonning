# train_bc.py — τελική έκδοση με επιλογή session, 120 epochs & ποσοστιαία metrics
import math
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

from dataset import SteeringDataset
from model import SmallCNN
from config import (DATA_ROOT, MODEL_PATH, DEVICE, BATCH_SIZE, LR, VAL_SPLIT)

# ======================= ΡΥΘΜΙΣΕΙΣ =======================
EPOCHS = 150
CLASS_THRESH = 0.15       # [-1,-0.15)=Left, [-0.15,+0.15]=Straight, (+0.15,1]=Right
NUM_WORKERS = 0           # Windows-safe
PRINT_EVERY = 1
MODEL_DIR = Path("models")

# 🔹 ΔΩΣΕ ΕΔΩ ΤΟ SESSION ΠΟΥ ΘΕΣ 🔹
SESSION_PATH = Path("data\session_2025-10-22_11-51-51_good")
# ==========================================================


def to_class_idx(y: torch.Tensor, thr: float = CLASS_THRESH) -> torch.Tensor:
    """Συνεχές steer ∈[-1,1] → 3 κλάσεις: 0=Left, 1=Straight, 2=Right"""
    y = y.view(-1)
    cls = torch.empty_like(y, dtype=torch.long)
    cls[y < -thr] = 0
    cls[(y >= -thr) & (y <= thr)] = 1
    cls[y > thr] = 2
    return cls


def accuracy_from_preds(pred: torch.Tensor, target: torch.Tensor, thr: float = CLASS_THRESH) -> float:
    pred_cls = to_class_idx(pred.detach().cpu(), thr)
    true_cls = to_class_idx(target.detach().cpu(), thr)
    return (pred_cls == true_cls).float().mean().item()


def confusion_matrix_3x3(model: nn.Module, loader: DataLoader, device: str, thr: float = CLASS_THRESH) -> np.ndarray:
    """Confusion matrix 3x3 σε validation. Rows=true, Cols=pred."""
    cm = np.zeros((3, 3), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred_cls = to_class_idx(pred.cpu(), thr).numpy()
            true_cls = to_class_idx(y.cpu(), thr).numpy()
            for t, p in zip(true_cls, pred_cls):
                cm[t, p] += 1
    return cm


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for imgs, steers in loader:
            imgs = imgs.to(DEVICE); steers = steers.to(DEVICE)
            preds = model(imgs)
            loss = F.smooth_l1_loss(preds, steers)
            mae = torch.mean(torch.abs(preds - steers))
            total_loss += loss.item() * imgs.size(0)
            total_mae  += mae.item()  * imgs.size(0)
            count += imgs.size(0)
    return total_loss / max(1,count), total_mae / max(1,count)


def main():
    print(">>> main() entry point reached <<<")
    print("[TRAIN] train_bc.py starting")

    sess_dir = Path(SESSION_PATH)
    csv_path = sess_dir / "labels.csv"
    img_dir = sess_dir / "images"

    if not csv_path.exists() or not img_dir.exists():
        print("[ERROR] Δεν βρέθηκαν labels.csv ή images/ στον φάκελο:", sess_dir)
        return

    print(f"[TRAIN] Using session: {sess_dir.name}")
    ds = SteeringDataset(csv_path)
    n = len(ds)
    print(f"[DATASET] Loaded {n} samples from CSV.")
    if n == 0:
        print("[ERROR] Το dataset είναι κενό.")
        return

    val_size = max(1, int(n * VAL_SPLIT))
    train_size = n - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"[TRAIN] Samples: total={n}, train={train_size}, val={val_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SmallCNN().to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    best_val = math.inf
    best_state = None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Όνομα αρχείου με βάση το session
    session_tag = sess_dir.name.replace("session_", "")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_best_path = MODEL_DIR / f"policy_{session_tag}_best.pt"
    model_latest_path = MODEL_DIR / "policy_latest.pt"

    for epoch in range(1, EPOCHS + 1):
        # --------- TRAIN ---------
        model.train()
        tot, loss_sum = 0, 0.0
        train_correct, train_total = 0, 0

        val_loss, val_mae = evaluate(model, val_loader)
        #print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mae={val_mae:.4f}")

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            tot += bs

            with torch.no_grad():
                acc_b = accuracy_from_preds(pred, y)
                train_correct += int(acc_b * bs)
                train_total += bs

        train_loss = loss_sum / max(1, tot)
        train_acc = train_correct / max(1, train_total)

        # --------- VALID ---------
        model.eval()
        vtot, vloss = 0, 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)
                bs = x.size(0)
                vloss += loss.item() * bs
                vtot += bs
                acc_b = accuracy_from_preds(pred, y)
                val_correct += int(acc_b * bs)
                val_total += bs

        val_loss = vloss / max(1, vtot)
        val_acc = val_correct / max(1, val_total)

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS:
            print(f"[EPOCH {epoch}/{EPOCHS}] "
                  f"train_loss={train_loss:.4f} | train_acc={format_pct(train_acc)} | "
                  f"val_loss={val_loss:.4f} | val_acc={format_pct(val_acc)}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            torch.save(best_state, model_best_path)
            torch.save(best_state, model_latest_path)
            print(f"[SAVE] Best model → {model_best_path.name} (val_loss={best_val:.4f})")

    print(f"\n[DONE] Training complete. Best validation loss: {best_val:.4f}")

    # -------- Confusion Matrix --------
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    cm = confusion_matrix_3x3(model, val_loader, DEVICE, thr=CLASS_THRESH)
    val_acc_final = cm.trace() / max(1, cm.sum())

    print("\n[METRICS] Validation accuracy (final/best):", format_pct(val_acc_final))
    print("[METRICS] Confusion Matrix (rows=true, cols=pred):")
    labels = ["Left", "Straight", "Right"]
    header = "             " + "  ".join([f"{c:^9}" for c in labels])
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>10} | " + "  ".join([f"{v:^9d}" for v in row]))

    print("\n[METRICS] Confusion Matrix normalized by row (%):")
    print(header)
    for i, row in enumerate(cm):
        s = row.sum() if row.sum() > 0 else 1
        pct_row = (row / s) * 100.0
        print(f"{labels[i]:>10} | " + "  ".join([f"{v:>8.1f}%" for v in pct_row]))


if __name__ == "__main__":
    main()
