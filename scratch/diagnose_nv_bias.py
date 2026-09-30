"""
Deep Diagnostic Audit: Testing NV Bias across all 7 HAM10000 classes and out-of-domain images.
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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import model
from models.efficientnet_b0_model import get_efficientnet_b0_model, get_efficientnet_b0_transform
from models.resnet50_model import get_resnet50_model, get_resnet50_transform
from models.inceptionv3_model import get_inceptionv3_model, get_inceptionv3_transform

MODELS_DIR = PROJECT_ROOT / "models"
CKPT_B0 = (MODELS_DIR / "efficientnet_b0_phase3_best.pth") if (MODELS_DIR / "efficientnet_b0_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "efficientnet_b0_phase3_best.pth")
CKPT_RESNET = (MODELS_DIR / "resnet50_phase3_best.pth") if (MODELS_DIR / "resnet50_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "resnet50_phase3_best.pth")
CKPT_INCEPTION = MODELS_DIR / "inceptionv3_candidate.pth"
METADATA_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_cache_256"
IMG_DIR_1 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_2"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def run_nv_bias_audit():
    print("=" * 80)
    print("DEEP DIAGNOSTIC AUDIT: TESTING CLASS PREDICTIONS ACROSS ALL 7 CLASSES")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    m_b0 = get_efficientnet_b0_model(num_classes=7, checkpoint_path=CKPT_B0, device=device, eval_mode=True)
    m_res = get_resnet50_model(num_classes=7, checkpoint_path=CKPT_RESNET, device=device, eval_mode=True)
    m_inc = get_inceptionv3_model(num_classes=7, checkpoint_path=CKPT_INCEPTION, device=device, eval_mode=True)

    tf_256 = get_efficientnet_b0_transform()
    tf_299 = get_inceptionv3_transform()

    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    # Test set split (seed 16)
    _, test_l = train_test_split(unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx'])
    test_df = df[df['lesion_id'].isin(test_l)].copy().reset_index(drop=True)

    # Sample 5 images per class (35 images total)
    sampled_dfs = []
    for cls in CLASS_NAMES:
        cls_sub = test_df[test_df['dx'] == cls]
        n_sample = min(5, len(cls_sub))
        sampled_dfs.append(cls_sub.sample(n=n_sample, random_state=42))

    eval_df = pd.concat(sampled_dfs).reset_index(drop=True)

    print(f"\nEvaluating {len(eval_df)} test set images (5 per class)...")
    print("=" * 95)
    print(f"{'#':<3} | {'True Class':<10} | {'B0 Pred':<10} | {'ResNet Pred':<10} | {'Inception Pred':<12} | {'Ensemble Pred':<12} | {'Status':<6}")
    print("=" * 95)

    class_correct = {c: 0 for c in CLASS_NAMES}
    class_total   = {c: 0 for c in CLASS_NAMES}
    pred_counts   = {c: 0 for c in CLASS_NAMES}

    for i, row in eval_df.iterrows():
        img_id = row['image_id']
        true_cls = row['dx']
        class_total[true_cls] += 1

        c_path = CACHE_DIR / f"{img_id}.jpg"
        if c_path.exists():
            img_path = c_path
        else:
            p1 = IMG_DIR_1 / f"{img_id}.jpg"
            p2 = IMG_DIR_2 / f"{img_id}.jpg"
            img_path = p1 if p1.exists() else p2

        img_obj = Image.open(img_path).convert("RGB")

        p_b0 = model.run_tta_inference(m_b0, tf_256(img_obj).unsqueeze(0))
        p_res = model.run_tta_inference(m_res, tf_256(img_obj).unsqueeze(0))
        p_inc = model.run_tta_inference(m_inc, tf_299(img_obj).unsqueeze(0))

        pred_b0_cls = CLASS_NAMES[int(np.argmax(p_b0))]
        pred_res_cls = CLASS_NAMES[int(np.argmax(p_res))]
        pred_inc_cls = CLASS_NAMES[int(np.argmax(p_inc))]

        ens_res = model.predict(img_obj, generate_cam=False)
        pred_ens_cls = ens_res['predicted_class']
        pred_counts[pred_ens_cls] += 1

        if pred_ens_cls == true_cls:
            class_correct[true_cls] += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"

        print(f"{i+1:<3} | {true_cls:<10} | {pred_b0_cls:<10} | {pred_res_cls:<10} | {pred_inc_cls:<12} | {pred_ens_cls:<12} | {status}")

    print("=" * 95)
    print("\nPer-Class Ensemble Test Performance:")
    for cls in CLASS_NAMES:
        corr = class_correct[cls]
        tot = class_total[cls]
        acc = (corr / tot * 100) if tot > 0 else 0
        print(f"  Class {cls.upper():<6} : {corr}/{tot} correct ({acc:.1f}%) | Total Times Predicted: {pred_counts[cls]}")

    # Also test test.webp if exists
    test_webp = PROJECT_ROOT / "test.webp"
    if test_webp.exists():
        print(f"\n--- Testing root test.webp ---")
        w_img = Image.open(test_webp).convert("RGB")
        w_res = model.predict(w_img, generate_cam=False)
        print(f"  test.webp Prediction: {w_res['predicted_class']} (Confidence: {w_res['confidence']*100:.2f}%)")
        print(f"  Probabilities: {w_res['probabilities']}")

if __name__ == "__main__":
    run_nv_bias_audit()
