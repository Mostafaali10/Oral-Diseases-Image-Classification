"""
Phase 4: Leakage-Resistant Splitting
- تقسيم على مستوى group_id
- Train 70% | Val 15% | Test 15%
"""

from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "final_manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
RANDOM_STATE = 42
# ====================================================

def copy_split(df, split_name):
    for label in df["label"].unique():
        (SPLITS_DIR / split_name / label).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        src = Path(row["path"])
        dst = SPLITS_DIR / split_name / row["label"] / row["filename"]
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    df = pd.read_csv(MANIFEST)
    print(f"Total images: {len(df)}")
    print(df["label"].value_counts())

    # ----- Test split (15%) -----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(df, groups=df["group_id"]))

    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # ----- Val split (≈15% من الأصلي) -----
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["group_id"]))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    print("\n===== Split sizes =====")
    print(f"Train : {len(train)}")
    print(f"Val   : {len(val)}")
    print(f"Test  : {len(test)}")

    print("\n===== Class distribution =====")
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n{name}:")
        print(part["label"].value_counts())

    # حفظ الـ manifests
    train.to_csv(MANIFESTS_DIR / "train.csv", index=False)
    val.to_csv(MANIFESTS_DIR / "val.csv", index=False)
    test.to_csv(MANIFESTS_DIR / "test.csv", index=False)

    # نسخ الصور
    copy_split(train, "train")
    copy_split(val, "val")
    copy_split(test, "test")

    print(f"\nSplits ready → {SPLITS_DIR}")


if __name__ == "__main__":
    main()