"""
Phase 3: Dataset Finalization
- يختار صورة واحدة فقط من كل group
- يعمل نسخة نهائية نظيفة في data/final
- يحفظ manifest كامل
"""

from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS_CSV = PROJECT_ROOT / "data" / "near_duplicate" / "near_duplicate_groups.csv"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
# ====================================================

def main():
    df = pd.read_csv(GROUPS_CSV)
    print(f"Loaded groups: {len(df)} rows | {df['group_id'].nunique()} unique groups")

    # نأخذ أول صورة فقط من كل group
    final_df = df.drop_duplicates(subset=["group_id"], keep="first").reset_index(drop=True)
    print(f"After keeping 1 image per group: {len(final_df)}")

    print("\nClass distribution (final):")
    print(final_df["label"].value_counts())

    # نسخ الصور
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for label in final_df["label"].unique():
        (FINAL_DIR / label).mkdir(exist_ok=True)

    records = []
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Copying"):
        src = Path(row["path"])
        dst = FINAL_DIR / row["label"] / row["filename"]

        if not dst.exists():
            shutil.copy2(src, dst)

        records.append({
            "path": str(dst),
            "label": row["label"],
            "filename": row["filename"],
            "group_id": row["group_id"],
            "phash": row["phash"],
            "status": "accepted"
        })

    manifest = pd.DataFrame(records)
    out_csv = MANIFESTS_DIR / "final_manifest.csv"
    manifest.to_csv(out_csv, index=False)

    print(f"\nFinal dataset ready → {FINAL_DIR}")
    print(f"Manifest saved   → {out_csv}")
    print(f"Total accepted images: {len(manifest)}")


if __name__ == "__main__":
    main()