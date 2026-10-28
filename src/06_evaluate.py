from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "resnet50_hparam_best.keras"
TEST = ROOT / "data" / "splits" / "test"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

if not MODEL.exists():
    raise FileNotFoundError(f"حط الموديل هنا الأول: {MODEL}")

print("Loading model:", MODEL.name)
model = keras.models.load_model(MODEL)
test_ds = keras.utils.image_dataset_from_directory(
    TEST, image_size=(224, 224), batch_size=32, label_mode="int", shuffle=False
)
class_names = test_ds.class_names
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
print("\nAccuracy", round(acc, 4), "Macro F1", round(f1, 4))
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

pd.DataFrame([{
    "Model": MODEL.stem,
    "Accuracy": round(acc, 4),
    "Macro_F1": round(f1, 4),
    "Macro_Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
    "Macro_Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
}]).to_csv(REPORTS / "hparam_test_results.csv", index=False)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(FIGURES / "confusion_matrix_hparam.png", dpi=150)
plt.close()
print("Saved", REPORTS / "hparam_test_results.csv")