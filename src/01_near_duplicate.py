"""
Phase 2: Near-Duplicate Detection using Perceptual Hash
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CLEAN = PROJECT_ROOT / "data" / "raw_clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "near_duplicate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HASH_SIZE = 16
THRESHOLD = 8          # المسافة المسموح بها
# ====================================================

def compute_phash(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return str(imagehash.phash(img, hash_size=HASH_SIZE))
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return None


def main():
    records = []
    class_dirs = sorted([d for d in RAW_CLEAN.iterdir() if d.is_dir()])

    print(">>> Computing perceptual hashes...")
    for class_dir in class_dirs:
        label = class_dir.name
        images = [p for p in class_dir.glob("*.*") 
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

        for img_path in tqdm(images, desc=label):
            ph = compute_phash(img_path)
            if ph is None:
                continue
            records.append({
                "path": str(img_path),
                "label": label,
                "filename": img_path.name,
                "phash": ph
            })

    df = pd.DataFrame(records)
    print(f"\nTotal images hashed: {len(df)}")

    # ---------- Grouping ----------
    print(">>> Grouping near-duplicates...")
    groups = []
    used = set()
    group_id = 0

    phashes = df["phash"].tolist()
    paths = df["path"].tolist()
    labels = df["label"].tolist()
    filenames = df["filename"].tolist()

    for i in tqdm(range(len(df))):
        if i in used:
            continue

        current_group = [i]
        used.add(i)
        h1 = imagehash.hex_to_hash(phashes[i])

        for j in range(i + 1, len(df)):
            if j in used:
                continue
            h2 = imagehash.hex_to_hash(phashes[j])
            if h1 - h2 <= THRESHOLD:
                current_group.append(j)
                used.add(j)

        for idx in current_group:
            groups.append({
                "group_id": group_id,
                "path": paths[idx],
                "label": labels[idx],
                "filename": filenames[idx],
                "phash": phashes[idx]
            })
        group_id += 1

    groups_df = pd.DataFrame(groups)
    out_csv = OUTPUT_DIR / "near_duplicate_groups.csv"
    groups_df.to_csv(out_csv, index=False)

    multi = groups_df.groupby("group_id").size()
    print(f"\nTotal groups     : {group_id}")
    print(f"Groups > 1 image : {(multi > 1).sum()}")
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()