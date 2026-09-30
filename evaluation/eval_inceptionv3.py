"""
Standalone Evaluation of InceptionV3 Candidate Model.
Computes validation and test metrics (Accuracy, Macro F1, ROC-AUC, Per-Class Breakdown, Confusion Matrix).
Saves results to: models/experiment_b_inceptionv3_metrics.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.inceptionv3_model import get_inceptionv3_model

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "inceptionv3_candidate.pth"
METRICS_OUT_PATH = MODELS_DIR / "experiment_b_inceptionv3_metrics.json"
METADATA_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_cache_256"
IMG_DIR_1 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_2"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INCEPTION_SIZE = 299

class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(img, padding, fill=0, padding_mode='edge')

def get_transforms():
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def evaluate_dataset(model, df, device):
    transform = get_transforms()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            img_id = row['image_id']
            c_path = CACHE_DIR / f"{img_id}.jpg"
            if c_path.exists():
                img_path = c_path
            else:
                p1 = IMG_DIR_1 / f"{img_id}.jpg"
                p2 = IMG_DIR_2 / f"{img_id}.jpg"
                img_path = p1 if p1.exists() else p2

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            label_idx = CLASS_TO_IDX[row['dx']]
            pred_idx = int(np.argmax(probs))

            y_true.append(label_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    try:
        roc_auc = float(roc_auc_score(y_true_oh, y_probs, average='macro', multi_class='ovr'))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    per_class_auc = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            per_class_auc[name] = float(roc_auc_score(y_true_oh[:, i], y_probs[:, i]))
        except Exception:
            per_class_auc[name] = 0.0

    per_class_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class_dict[name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "auc_roc": round(float(per_class_auc[name]), 4)
        }

    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
        "macro_roc_auc": round(float(roc_auc), 4),
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm
    }

def run_eval_inceptionv3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running InceptionV3 Standalone Evaluation on device: {device}")

    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    train_val_l, test_l = train_test_split(unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx'])
    train_val_df = df[df['lesion_id'].isin(train_val_l)]
    train_val_unique = train_val_df.drop_duplicates(subset='lesion_id')

    train_l, val_l = train_test_split(train_val_unique['lesion_id'], test_size=0.17647, random_state=30, stratify=train_val_unique['dx'])
    val_df = df[df['lesion_id'].isin(val_l)].copy()
    test_df = df[df['lesion_id'].isin(test_l)].copy()

    model = get_inceptionv3_model(num_classes=NUM_CLASSES, checkpoint_path=CHECKPOINT_PATH, device=device, eval_mode=True)

    print("Evaluating InceptionV3 on Validation set...")
    val_results = evaluate_dataset(model, val_df, device)
    print(f"Validation Accuracy: {val_results['accuracy']*100:.2f}% | Macro F1: {val_results['macro_f1']:.4f}")

    print("Evaluating InceptionV3 on Test set (1,484 images)...")
    test_results = evaluate_dataset(model, test_df, device)
    print(f"Test Accuracy: {test_results['accuracy']*100:.2f}% | Macro F1: {test_results['macro_f1']:.4f} | ROC-AUC: {test_results['macro_roc_auc']:.4f}")

    metrics_output = {
        "model_name": "InceptionV3 Candidate Model (Experiment B - Class-Weighted Loss)",
        "validation_metrics": val_results,
        "test_metrics": test_results
    }

    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"InceptionV3 Standalone evaluation metrics saved to: {METRICS_OUT_PATH}")

if __name__ == "__main__":
    run_eval_inceptionv3()
