"""
Three-Model Ensemble (EfficientNet-B0 + ResNet-50 + InceptionV3) Evaluation Pipeline.

Pipeline Steps:
1. Validation Set Grid Search: Evaluate probability fusion across 7 candidate weight ratios ONLY on Validation set.
2. Freeze Best Fusion Weights based on validation performance.
3. Apply Existing 5-View TTA across all 3 models.
4. Single Final Evaluation on 1,484 Untouched Test Set Images.
5. Model Selection Decision: Compare against 84.30% Baseline.
6. Export detailed metrics JSON to models/ensemble_b0_resnet50_inceptionv3_metrics.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import time
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

from models.efficientnet_b0_model import get_efficientnet_b0_model
from models.resnet50_model import get_resnet50_model
from models.inceptionv3_model import get_inceptionv3_model

MODELS_DIR = PROJECT_ROOT / "models"
CKPT_B0 = (MODELS_DIR / "efficientnet_b0_phase3_best.pth") if (MODELS_DIR / "efficientnet_b0_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "efficientnet_b0_phase3_best.pth")
CKPT_RESNET = (MODELS_DIR / "resnet50_phase3_best.pth") if (MODELS_DIR / "resnet50_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "resnet50_phase3_best.pth")
CKPT_INCEPTION = MODELS_DIR / "inceptionv3_candidate.pth"
OUT_METRICS = MODELS_DIR / "ensemble_b0_resnet50_inceptionv3_metrics.json"

METADATA_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_cache_256"
IMG_DIR_1 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_2"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Candidate weight ratios for (B0, ResNet50, InceptionV3)
CANDIDATE_WEIGHTS = [
    (0.50, 0.30, 0.20),
    (0.50, 0.25, 0.25),
    (0.45, 0.30, 0.25),
    (0.45, 0.25, 0.30),
    (0.40, 0.30, 0.30),
    (0.40, 0.35, 0.25),
    (0.35, 0.35, 0.30),
]

class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(img, padding, fill=0, padding_mode='edge')

def get_base_transform(img_size):
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

def get_tta_views(raw_tensor):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    v0 = norm(raw_tensor)
    v1 = norm(TF.hflip(raw_tensor))
    v2 = norm(TF.vflip(raw_tensor))
    v3 = norm(TF.rotate(raw_tensor, 15))
    v4 = norm(TF.rotate(raw_tensor, -15))
    return [v0, v1, v2, v3, v4]

def load_all_models(device):
    print("Loading EfficientNet-B0 model checkpoint...")
    b0 = get_efficientnet_b0_model(num_classes=NUM_CLASSES, checkpoint_path=CKPT_B0, device=device, eval_mode=True)

    print("Loading ResNet-50 model checkpoint...")
    resnet = get_resnet50_model(num_classes=NUM_CLASSES, checkpoint_path=CKPT_RESNET, device=device, eval_mode=True)

    print("Loading InceptionV3 candidate model checkpoint...")
    inc = get_inceptionv3_model(num_classes=NUM_CLASSES, checkpoint_path=CKPT_INCEPTION, device=device, eval_mode=True)

    return b0, resnet, inc

def predict_single_model_tta(model, raw_tensor, device):
    views = get_tta_views(raw_tensor)
    view_probs = []
    with torch.no_grad():
        for view in views:
            out = model(view.to(device))
            p = torch.softmax(out, dim=1)[0].cpu().numpy()
            view_probs.append(p)
    return np.mean(view_probs, axis=0)

def extract_dataset_predictions(b0, resnet, inc, df, device):
    tf_256 = get_base_transform(256)
    tf_299 = get_base_transform(299)

    probs_b0, probs_resnet, probs_inc, y_true = [], [], [], []

    print(f"Extracting TTA prediction probabilities for {len(df)} images...")
    t0 = time.time()
    for idx, (_, row) in enumerate(df.iterrows()):
        img_id = row['image_id']
        c_path = CACHE_DIR / f"{img_id}.jpg"
        if c_path.exists():
            img_path = c_path
        else:
            p1 = IMG_DIR_1 / f"{img_id}.jpg"
            p2 = IMG_DIR_2 / f"{img_id}.jpg"
            img_path = p1 if p1.exists() else p2

        img = Image.open(img_path).convert("RGB")
        tensor_256 = tf_256(img).unsqueeze(0)
        tensor_299 = tf_299(img).unsqueeze(0)

        p_b0 = predict_single_model_tta(b0, tensor_256, device)
        p_res = predict_single_model_tta(resnet, tensor_256, device)
        p_inc = predict_single_model_tta(inc, tensor_299, device)

        probs_b0.append(p_b0)
        probs_resnet.append(p_res)
        probs_inc.append(p_inc)
        y_true.append(CLASS_TO_IDX[row['dx']])

        if (idx + 1) % 300 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} images ({time.time() - t0:.1f}s)")

    return np.array(probs_b0), np.array(probs_resnet), np.array(probs_inc), np.array(y_true)

def compute_metrics(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    try:
        roc_auc = float(roc_auc_score(y_true_oh, y_probs, average='macro', multi_class='ovr'))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    per_p, per_r, per_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            auc = float(roc_auc_score(y_true_oh[:, i], y_probs[:, i]))
        except Exception:
            auc = 0.0
        per_class_dict[name] = {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1_score": round(float(per_f1[i]), 4),
            "auc_roc": round(auc, 4)
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

def run_ensemble_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("3-MODEL ENSEMBLE EVALUATION PIPELINE")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Metadata and Create Splits
    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    train_val_l, test_l = train_test_split(unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx'])
    train_val_df = df[df['lesion_id'].isin(train_val_l)]
    train_val_unique = train_val_df.drop_duplicates(subset='lesion_id')

    train_l, val_l = train_test_split(train_val_unique['lesion_id'], test_size=0.17647, random_state=30, stratify=train_val_unique['dx'])
    val_df = df[df['lesion_id'].isin(val_l)].copy()
    test_df = df[df['lesion_id'].isin(test_l)].copy()

    print(f"Dataset Splits Verified: Val={len(val_df)}, Test={len(test_df)}")

    # 2. Load Models
    b0, resnet, inc = load_all_models(device)

    # 3. Extract Validation Predictions
    print("\n--- Phase 1: Extracting Validation Prediction Probabilities ---")
    val_p_b0, val_p_res, val_p_inc, val_y = extract_dataset_predictions(b0, resnet, inc, val_df, device)

    # Grid Search Fusion Weights on Validation Set
    print("\n--- Phase 2: Grid Searching Fusion Weights on Validation Set ---")
    best_val_acc = -1.0
    best_weights = None
    grid_results = []

    for w_b0, w_res, w_inc in CANDIDATE_WEIGHTS:
        val_ens_probs = w_b0 * val_p_b0 + w_res * val_p_res + w_inc * val_p_inc
        m = compute_metrics(val_y, val_ens_probs)
        acc = m["accuracy"]
        f1 = m["macro_f1"]
        grid_results.append({
            "weights_ratio": [w_b0, w_res, w_inc],
            "val_accuracy": acc,
            "val_macro_f1": f1
        })
        print(f"  Weights ({w_b0:.2f}, {w_res:.2f}, {w_inc:.2f}) -> Val Acc: {acc*100:.2f}% | Val Macro F1: {f1:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            best_weights = (w_b0, w_res, w_inc)

    print(f"\nOptimal Validation Weights Selected: {best_weights} with Val Acc: {best_val_acc*100:.2f}%")

    # 4. Extract Test Set Predictions & Evaluate Single Final Metric
    print("\n--- Phase 3: Extracting Test Set Prediction Probabilities ---")
    test_p_b0, test_p_res, test_p_inc, test_y = extract_dataset_predictions(b0, resnet, inc, test_df, device)

    w_b0, w_res, w_inc = best_weights
    test_ens_probs = w_b0 * test_p_b0 + w_res * test_p_res + w_inc * test_p_inc
    test_metrics = compute_metrics(test_y, test_ens_probs)
    test_acc = test_metrics["accuracy"]

    print("\n" + "=" * 80)
    print("FINAL UNTOUCHED TEST SET EVALUATION (1,484 Images)")
    print("=" * 80)
    print(f"3-Model Ensemble Test Accuracy : {test_acc*100:.2f}%")
    print(f"Macro F1 Score                  : {test_metrics['macro_f1']:.4f}")
    print(f"Macro Precision                 : {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall                    : {test_metrics['macro_recall']:.4f}")
    print(f"Macro ROC-AUC                   : {test_metrics['macro_roc_auc']:.4f}")

    baseline_acc = 0.842992
    diff = (test_acc - baseline_acc) * 100
    is_superior = test_acc > baseline_acc

    if is_superior:
        rec = "RECOMMEND_NEW_CANDIDATE_MODEL"
        reason = f"The 3-model ensemble achieved {test_acc*100:.2f}% test accuracy, outperforming the current baseline of {baseline_acc*100:.2f}% by +{diff:.2f}%."
    else:
        rec = "KEEP_EXISTING_PRODUCTION_MODEL"
        reason = f"The 3-model ensemble achieved {test_acc*100:.2f}% test accuracy, which did not exceed baseline {baseline_acc*100:.2f}%."

    print(f"\nDecision Recommendation : {rec}")
    print(f"Reasoning               : {reason}")

    out_data = {
        "experiment_name": "EfficientNet-B0 + ResNet-50 + InceptionV3 3-Model Ensemble",
        "baseline_test_accuracy": round(baseline_acc, 6),
        "new_ensemble_test_accuracy": round(test_acc, 6),
        "improvement_over_baseline": f"{'+' if diff >= 0 else ''}{diff:.2f}%",
        "is_superior_to_baseline": is_superior,
        "official_model_recommendation": rec,
        "decision_reasoning": reason,
        "selected_fusion_weights": {
            "efficientnet_b0": w_b0,
            "resnet_50": w_res,
            "inception_v3": w_inc
        },
        "validation_weight_search_grid": grid_results,
        "final_test_metrics": test_metrics
    }

    with open(OUT_METRICS, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nMetrics JSON report saved to: {OUT_METRICS}")

if __name__ == "__main__":
    run_ensemble_evaluation()
