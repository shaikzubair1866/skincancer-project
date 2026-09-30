"""
Step 10 Verification Script — Testing 5 Genuine HAM10000 Images
Produces exact report format required by Step 10 & Step 13.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import model
from models.efficientnet_b0_model import get_efficientnet_b0_transform

METADATA_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_cache_256"
IMG_DIR_1 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_2"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
TARGET_5_CLASSES = ["akiec", "bcc", "bkl", "mel", "vasc"]

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

def run_step_10_test():
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file missing at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    _, test_l = train_test_split(
        unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx']
    )
    test_df = df[df['lesion_id'].isin(test_l)].copy().reset_index(drop=True)

    selected_images = []
    for cls in TARGET_5_CLASSES:
        cls_sub = test_df[test_df['dx'] == cls]
        for _, row in cls_sub.iterrows():
            imgPath = find_image_path(row['image_id'])
            if imgPath is not None:
                selected_images.append((cls, row['image_id'], imgPath))
                break

    print("=" * 80)
    print("STEP 10 & STEP 13 — 5-IMAGE DIAGNOSTIC REPORT")
    print("=" * 80)

    for true_cls, img_id, img_path in selected_images:
        pil_img = Image.open(img_path).convert("RGB")
        arr = np.array(pil_img)

        diag = model.predict_debug(pil_img)

        b0_info = diag["efficientnet_prediction"]
        res_info = diag["resnet_prediction"]
        inc_info = diag["inception_prediction"]
        ens_info = diag["ensemble_prediction"]

        b0_probs = [round(float(v), 6) for v in b0_info["probabilities"].values()]
        res_probs = [round(float(v), 6) for v in res_info["probabilities"].values()]
        inc_probs = [round(float(v), 6) for v in inc_info["probabilities"].values()]
        ens_probs = [round(float(v), 6) for v in ens_info["probabilities"].values()]

        print("\nIMAGE:")
        print(f"filename: {img_path.name} (True Class: {true_cls.upper()})")

        print("\nINPUT:")
        print(f"size: {pil_img.size}")
        print(f"mean: {arr.mean():.4f}")
        print(f"std: {arr.std():.4f}")

        print("\nEFFICIENTNET:")
        print(f"probabilities: {b0_probs}")
        print(f"prediction: {b0_info['class']}")

        print("\nRESNET:")
        print(f"probabilities: {res_probs}")
        print(f"prediction: {res_info['class']}")

        print("\nINCEPTION:")
        print(f"probabilities: {inc_probs}")
        print(f"prediction: {inc_info['class']}")

        print("\nENSEMBLE:")
        print(f"probabilities: {ens_probs}")
        print(f"prediction: {ens_info['class']}")
        print(f"confidence: {ens_info['confidence']:.6f}")
        print("-" * 60)

    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_step_10_test()
