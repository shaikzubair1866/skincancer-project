"""
Debug Prediction Tool for Skin Cancer Classification Ensemble.
Runs diagnostic inference across B0, ResNet-50, InceptionV3, and Ensemble for 5+ images of different classes,
calculates tensor statistics min/max/mean/std for completely different images, and checks for identical probability vectors.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import model
from models.efficientnet_b0_model import get_efficientnet_b0_transform
from models.inceptionv3_model import get_inceptionv3_transform

METADATA_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_cache_256"
IMG_DIR_1 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_2"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def find_image_path(img_id):
    c_path = CACHE_DIR / f"{img_id}.jpg"
    if c_path.exists():
        return c_path
    p1 = IMG_DIR_1 / f"{img_id}.jpg"
    if p1.exists():
        return p1
    p2 = IMG_DIR_2 / f"{img_id}.jpg"
    if p2.exists():
        return p2
    return None

def task_6_tensor_statistics_test(img_a_path, img_b_path):
    print("\n" + "=" * 70)
    print("TASK 6 — PREPROCESSING / TENSOR DIFFERENCE TEST")
    print("=" * 70)

    tf_256 = get_efficientnet_b0_transform()

    img_a = Image.open(img_a_path).convert("RGB")
    img_b = Image.open(img_b_path).convert("RGB")

    t_a = tf_256(img_a)
    t_b = tf_256(img_b)

    print(f"Image A ({img_a_path.name}) tensor stats:")
    print(f"  Min:  {t_a.min().item():.6f}")
    print(f"  Max:  {t_a.max().item():.6f}")
    print(f"  Mean: {t_a.mean().item():.6f}")
    print(f"  Std:  {t_a.std().item():.6f}")

    print(f"\nImage B ({img_b_path.name}) tensor stats:")
    print(f"  Min:  {t_b.min().item():.6f}")
    print(f"  Max:  {t_b.max().item():.6f}")
    print(f"  Mean: {t_b.mean().item():.6f}")
    print(f"  Std:  {t_b.std().item():.6f}")

    diff = torch.abs(t_a - t_b).mean().item()
    print(f"\nMean Absolute Tensor Difference |A - B|: {diff:.6f}")
    if diff == 0:
        print("[CRITICAL ERROR] Image A and Image B produced IDENTICAL tensors! Upload/Preprocessing pipeline is BROKEN.")
    else:
        print("[PASS] Tensors A and B are DISTINCT. Preprocessing pipeline is working correctly.")
    print("=" * 70 + "\n")

def run_task_4_5_image_test():
    if not METADATA_PATH.exists():
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    _, test_l = train_test_split(
        unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx']
    )
    test_df = df[df['lesion_id'].isin(test_l)].copy().reset_index(drop=True)

    print("Selecting 7 untouched test images (1 per class) from HAM10000 dataset...")
    test_images = []
    for cls in CLASS_NAMES:
        cls_sub = test_df[test_df['dx'] == cls]
        for _, row in cls_sub.iterrows():
            imgPath = find_image_path(row['image_id'])
            if imgPath is not None:
                test_images.append((cls, row['image_id'], imgPath))
                break

    all_prob_vectors = []
    image_paths_tested = []

    print("\n" + "=" * 80)
    print("TASK 4 — MULTI-IMAGE PREDICTION AUDIT (7 DIFFERENT CLASSES)")
    print("=" * 80)

    for true_cls, img_id, img_path in test_images:
        image_paths_tested.append(img_path)
        img_obj = Image.open(img_path).convert("RGB")
        
        diag = model.predict_debug(img_obj)

        b0_info = diag["efficientnet_prediction"]
        res_info = diag["resnet_prediction"]
        inc_info = diag["inception_prediction"]
        ens_info = diag["ensemble_prediction"]

        ens_vector = list(ens_info["probabilities"].values())
        all_prob_vectors.append(ens_vector)

        print(f"\nFile: {img_path.name} | True Class: {true_cls.upper()}")
        print(f"  EfficientNet-B0 Pred : {b0_info['class']} (Conf: {b0_info['confidence']:.4f})")
        print(f"    Probs: {[round(v, 4) for v in b0_info['probabilities'].values()]}")
        print(f"  ResNet-50 Pred       : {res_info['class']} (Conf: {res_info['confidence']:.4f})")
        print(f"    Probs: {[round(v, 4) for v in res_info['probabilities'].values()]}")
        print(f"  InceptionV3 Pred     : {inc_info['class']} (Conf: {inc_info['confidence']:.4f})")
        print(f"    Probs: {[round(v, 4) for v in inc_info['probabilities'].values()]}")
        print(f"  Ensemble Pred        : {ens_info['class']} (Conf: {ens_info['confidence']:.4f})")
        print(f"    Probs: {[round(v, 4) for v in ens_vector]}")

    # Check for identical probability vectors
    print("\n" + "=" * 80)
    print("PROBABILITY VECTOR DIVERSITY CHECK")
    print("=" * 80)
    identical_found = False
    for i in range(len(all_prob_vectors)):
        for j in range(i + 1, len(all_prob_vectors)):
            v_i = np.array(all_prob_vectors[i])
            v_j = np.array(all_prob_vectors[j])
            if np.allclose(v_i, v_j, atol=1e-5):
                print(f"[ERROR] Image {i} ({test_images[i][1]}) and Image {j} ({test_images[j][1]}) produced IDENTICAL probability vectors!")
                identical_found = True

    if not identical_found:
        print("[PASS] All 7 images produced DISTINCT probability vectors and DIFFERENT class predictions!")
    print("=" * 80 + "\n")

    # Run Task 6 Tensor Stats Test on first two images
    if len(image_paths_tested) >= 2:
        task_6_tensor_statistics_test(image_paths_tested[0], image_paths_tested[1])

if __name__ == "__main__":
    run_task_4_5_image_test()
