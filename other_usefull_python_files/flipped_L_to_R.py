import cv2
from pathlib import Path

# ==============================
#  CONFIGURATION
# ==============================
# 🔹 Φάκελος εισόδου (όπου είναι οι εικόνες)
INPUT_DIR = r"C:\Users\nikpa\PycharmProjects\tello_line_following\data\session_2025-10-22_11-51-51_good\flipped_output"

# 🔹 Δημιουργεί αυτόματα φάκελο εξόδου
OUTPUT_DIR = Path(INPUT_DIR).parent / "flipped_horizontal"

# ==============================
#  MAIN CODE
# ==============================
def flip_all_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in input_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    print(f"[INFO] Found {len(images)} images in '{input_dir}'")

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable file: {img_path}")
            continue

        # 🔁 Flip οριζόντια (left ↔ right)
        flipped = cv2.flip(img, 1)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), flipped)
        print(f"[{i}/{len(images)}] Saved flipped: {out_path}")

    print(f"\n✅ Done! All images flipped horizontally (right↔left)!")
    print(f"[INFO] Output folder: {output_dir}")


if __name__ == "__main__":
    flip_all_images(INPUT_DIR, OUTPUT_DIR)
