from pathlib import Path
from datetime import datetime

# ----------------------
# Paths & Sessions
# ----------------------
DATA_ROOT = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Αυτόματο session ανά εκτέλεση (π.χ. session_2025-10-14_21-05-33)
SESS_NAME = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SESSION_DIR = DATA_ROOT / SESS_NAME
SESSION_DIR.mkdir(parents=True, exist_ok=True)

def latest_session_dir() -> Path | None:
    if not DATA_ROOT.exists():
        return None
    sessions = [p for p in DATA_ROOT.glob("session_*") if p.is_dir()]
    if not sessions:
        return None
    sessions.sort(key=lambda p: p.stat().st_mtime)
    return sessions[-1]

# ----------------------
# Model / Training
# ----------------------
MODEL_PATH = MODELS_DIR / "policy.pt"
DEVICE = "cpu"          # "cuda" αν έχεις NVIDIA + PyTorch CUDA
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.15

# ----------------------
# Image sizes (CNN input)
# ----------------------
IMG_W, IMG_H = 160, 120

# --- Flight constants ---
FORWARD_SPEED = 10
YAW_GAIN = 1.0         # ↑ πιο “ζωντανό” στρίψιμο (ήταν 0.6)
MAX_YAW_RC = 80        # ↑ ανώτατο yaw (ήταν 60)

# --- Stream and camera ---
TELLO_STREAM_URL = "udp://0.0.0.0:11111"
STREAM_WARMUP_SEC = 2.0

# --- Altitude control ---
ALT_TARGET_CM = 90    # σταθερό 90cm
ALT_KP = 0.8
ALT_UD_LIMIT = 15

# --- Misc ---
LOG_INTERVAL = 0.5
DEBUG = True


# ================== YAW CONTROL (FINAL) ==================
# Εδώ ορίζονται τα gain και όρια για πραγματικό στρίψιμο (yaw)

# Αν δεις ότι στρίβει ΑΝΑΠΟΔΑ (δεξιά αντί αριστερά), άλλαξέ το σε True
INVERT_YAW = False

# Νεκρή ζώνη στο steer για σταθερότητα
STEER_DEADBAND = 0.04

# Ελάχιστη τιμή για να ξεκινήσει η στροφή
MIN_YAW_RC = 28

# Κέρδος yaw και μέγιστο RC
YAW_GAIN   = 2.5          # δύναμη στροφής (δοκίμασε 2.0–3.0)
MAX_YAW_RC = 85           # μέγιστο yaw RC [-100..100]

# Για reference (παραμένει off)
USE_STRAFE     = False
MAX_STRAFE_RC  = 40

# Προώθηση — αρκετά μικρή ώστε να προλαβαίνει να στρίψει
FORWARD_SPEED = 10

# Όταν το λάθος είναι μεγάλο, κόψε προώθηση
CUT_FWD_ERR = 0.55


# ----------------------
# Video stream (OpenCV via FFmpeg)
# ----------------------
TELLO_STREAM_URL = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000"
STREAM_WARMUP_SEC = 2.0

