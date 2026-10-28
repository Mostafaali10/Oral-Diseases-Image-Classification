"""
Phase 5: Class Balancing via Augmentation (Train only)
- نوصل كل الكلاسات لنفس العدد تقريباً (أكبر كلاس)
- Augmentation على Train فقط
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from tqdm import tqdm
import cv2

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "splits" / "train"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
TARGET_COUNT = None          # لو None → ياخد أكبر كلاس
# ====================================================

def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.3),
    ])


def main():
    # حساب عدد الصور الحالي في كل كلاس
    class_counts = {}
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            imgs = list(class_dir.glob("*.*"))
            class_counts[class_dir.name] = len(imgs)

    print("Current Train distribution:")
    for k, v in class_counts.items():
        print(f"  {k:25s}: {v}")

    max_count = max(class_counts.values()) if TARGET_COUNT is None else TARGET_COUNT
    print(f"\nTarget per class: {max_count}")

    transform = get_augmentation()
    new_records = []

    for label, current in class_counts.items():
        needed = max_count - current
        if needed <= 0:
            print(f"{label}: already enough ({current})")
            continue

        print(f"\n{label}: generating {needed} augmented images...")
        class_path = TRAIN_DIR / label
        images = list(class_path.glob("*.*"))

        generated = 0
        idx = 0
        while generated < needed:
            src = images[idx % len(images)]
            img = cv2.imread(str(src))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            aug = transform(image=img)["image"]
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

            new_name = f"aug_{generated:04d}_{src.name}"
            dst = class_path / new_name
            cv2.imwrite(str(dst), aug_bgr)

            new_records.append({
                "path": str(dst),
                "label": label,
                "filename": new_name,
                "source_original": src.name,
                "type": "augmented"
            })
            generated += 1
            idx += 1

    # حفظ سجل الـ augmentation
    if new_records:
        aug_df = pd.DataFrame(new_records)
        aug_df.to_csv(MANIFESTS_DIR / "train_augmented.csv", index=False)
        print(f"\nAugmented images log → {MANIFESTS_DIR / 'train_augmented.csv'}")

    # توزيع نهائي
    print("\n===== Final Train distribution =====")
    for class_dir in sorted(TRAIN_DIR.iterdir()):
        if class_dir.is_dir():
            print(f"{class_dir.name:25s}: {len(list(class_dir.glob('*.*')))}")


if __name__ == "__main__":
    main()