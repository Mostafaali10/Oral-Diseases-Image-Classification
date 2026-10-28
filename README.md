# OralVision AI

Image classification for six oral conditions. The deployed model is a fine-tuned ResNet50 selected by validation macro F1, then measured once on a held-out test set.

**Test accuracy: 90.81%**  
**Test macro F1: 0.8467**

These numbers are from the leakage-resistant split in this repository. They are not comparable to an earlier 95% figure that used a different split.

## Classes

| Index | Class |
|---|---|
| 0 | Calculus |
| 1 | Caries |
| 2 | Gingivitis |
| 3 | Hypodontia |
| 4 | Mouth_Ulcer |
| 5 | Tooth_Discoloration |

The index order is the alphabetical order produced by `image_dataset_from_directory`. Do not reorder it.

## What the system does

1. Removes exact duplicates, then groups near-duplicates with perceptual hashing.
2. Keeps one image from each near-duplicate group (3,119 images).
3. Splits by group id so related images stay in the same split: train 2,183, validation 468, test 468.
4. Balances the **train** split only, by augmentation, to 777 images per class (4,662 files). Validation and test stay untouched.
5. Trains a custom CNN and three ImageNet models, fine-tunes the best two, then runs a small hyperparameter search.
6. Serves the winning checkpoint in a Flask app.

Source data: [Oral Diseases on Kaggle](https://www.kaggle.com/datasets/salmansajid05/oral-diseases). The original download is not modified. All cleaning writes new folders.

## Results

Validation accuracy during model selection:

| Model | Validation accuracy |
|---|---|
| Custom CNN | 0.791 |
| MobileNetV2 (frozen backbone) | 0.560 |
| EfficientNetB0 after fine-tuning | 0.906 |
| ResNet50 after fine-tuning | 0.917 |
| ResNet50 after hyperparameter search | 0.923 |

Hyperparameter search (train and validation only), ranked by validation macro F1:

| Trial | Learning rate | Backbone unfrozen | Val accuracy | Val macro F1 |
|---|---|---|---|---|
| lr1e-5_unfreeze25 | 1e-5 | 25% | 0.9231 | 0.8683 |
| lr1e-5_unfreeze10 | 1e-5 | 10% | 0.9209 | 0.8659 |
| lr5e-6_unfreeze25 | 5e-6 | 25% | 0.9231 | 0.8626 |
| lr1e-4_head_only | 1e-4 | head only | 0.9188 | 0.8505 |

Held-out test result for `models/resnet50_hparam_best.keras` (468 images, evaluated once):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calculus | 0.8214 | 0.9079 | 0.8625 | 76 |
| Caries | 0.9130 | 0.5833 | 0.7119 | 36 |
| Gingivitis | 0.8954 | 0.9786 | 0.9352 | 140 |
| Hypodontia | 0.9801 | 0.9673 | 0.9737 | 153 |
| Mouth_Ulcer | 0.9211 | 1.0000 | 0.9589 | 35 |
| Tooth_Discoloration | 0.7895 | 0.5357 | 0.6383 | 28 |

ResNet50 won because a pretrained backbone transfers better than a small CNN on this dataset. MobileNetV2 with a frozen backbone was not competitive. The hyperparameter step added about 0.6 accuracy points on the test set over the earlier fine-tuned ResNet50 (90.17% accuracy, 0.8461 macro F1). Caries and tooth discoloration stay weak: few real images, and they are easy to confuse with other classes. Augmentation balanced the training counts; it does not add new patients.

## Project layout

```text
OralVision-AI/
├── app/
│   ├── app.py                 # Flask UI and POST /predict
│   └── templates/index.html
├── data/
│   ├── raw_clean/             # 3,507 images after exact-hash cleaning
│   ├── final/                 # 3,119 images, one per near-duplicate group
│   ├── splits/train|val|test/
│   ├── manifests/
│   └── near_duplicate/
├── models/
│   └── resnet50_hparam_best.keras
├── reports/
│   ├── hyperparameter_trials.csv
│   ├── hparam_test_results.csv
│   └── figures/
└── src/
    ├── 01_near_duplicate.py
    ├── 02_finalize_dataset.py
    ├── 03_leakage_split.py
    ├── 04_balance_train.py
    ├── 05_train_models.py
    ├── 06_evaluate.py
    └── 08_hyperparameter_tuning.py
```

Training ran on Kaggle (Tesla T4). TensorFlow on native Windows does not use an NVIDIA GPU for TensorFlow 2.11+, so local runs are for evaluation and the web app.

## Run the app

Use the same Python environment that already has TensorFlow, not an empty virtualenv.

```powershell
cd H:\Week2\Sprint2\OralVision-AI
pip install -r requirements.txt
python app/app.py
```

Open http://127.0.0.1:5000

Upload a clear mouth image. The page shows the predicted class and the probability of each class. `POST /predict` with a form field named `file` returns JSON.

Images are passed as 224×224 RGB with pixel values in 0–255, matching training. Do not apply an extra rescale or `preprocess_input` in front of this checkpoint.

## Reproduce the evaluation

```powershell
python src/06b_eval_hparam.py
```

Only if that script is present and points at `models/resnet50_hparam_best.keras`. Do not use the test split to choose another model.

## Requirements covered

- Custom CNN
- Three pretrained models: MobileNetV2, EfficientNetB0, ResNet50
- Hyperparameter tuning on learning rate and how much of the backbone is trainable
- Comparison and discussion of the results
- Deployment with Flask, using the best checkpoint
