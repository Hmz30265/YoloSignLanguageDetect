import os
import random
import shutil
from pathlib import Path

# === Change this if your root folder is different ===
root_dir = Path("data")

# Create YOLO folder structure
for split in ["train", "val", "test"]:
    (root_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (root_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Find all images with corresponding .txt labels
image_exts = [".jpg", ".jpeg", ".png"]
all_pairs = []

for class_folder in root_dir.iterdir():
    if class_folder.is_dir() and class_folder.name not in {"images", "labels"}:
        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() in image_exts:
                label_path = img_path.with_suffix(".txt")
                if label_path.exists():
                    all_pairs.append((img_path, label_path))

# Shuffle and split
random.shuffle(all_pairs)
total = len(all_pairs)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

splits = {
    "train": all_pairs[:train_end],
    "val": all_pairs[train_end:val_end],
    "test": all_pairs[val_end:]
}

# Copy files into structure
for split, pairs in splits.items():
    for img, label in pairs:
        shutil.copy(img, root_dir / "images" / split / img.name)
        shutil.copy(label, root_dir / "labels" / split / label.name)

print("✅ Dataset split complete!")
print(f"Total images: {total}")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
