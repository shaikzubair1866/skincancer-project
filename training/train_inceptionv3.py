import os
import sys
import time
from pathlib import Path

_dll_path = r"C:\Users\isayy\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\lib"
if os.path.exists(_dll_path):
    os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_dll_path)
    except Exception:
        pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.inceptionv3_model import get_inceptionv3_model

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "inceptionv3_candidate.pth"
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

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()

class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(img, padding, fill=0, padding_mode='edge')

def get_data_loaders(batch_size=64):
    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    train_val_l, test_l = train_test_split(unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx'])
    train_val_df = df[df['lesion_id'].isin(train_val_l)]
    train_val_unique = train_val_df.drop_duplicates(subset='lesion_id')

    train_l, val_l = train_test_split(train_val_unique['lesion_id'], test_size=0.17647, random_state=30, stratify=train_val_unique['dx'])
    train_df = df[df['lesion_id'].isin(train_l)].copy()
    val_df = df[df['lesion_id'].isin(val_l)].copy()
    test_df = df[df['lesion_id'].isin(test_l)].copy()

    print_flush(f"InceptionV3 Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Untouched)")

    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

class HAM10000Dataset(torch.utils.data.Dataset):
    """Dynamic dataset loading images on-the-fly to prevent memory allocation crashes."""
    def __init__(self, df_subset, transform=None):
        self.df_subset = df_subset.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df_subset)

    def __getitem__(self, idx):
        row = self.df_subset.iloc[idx]
        img_id = row['image_id']
        c_path = CACHE_DIR / f"{img_id}.jpg"
        if c_path.exists():
            img_path = c_path
        else:
            p1 = IMG_DIR_1 / f"{img_id}.jpg"
            p2 = IMG_DIR_2 / f"{img_id}.jpg"
            img_path = p1 if p1.exists() else p2

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = CLASS_TO_IDX[row['dx']]
        return img, label

def get_data_loaders(batch_size=64):
    df = pd.read_csv(METADATA_PATH)
    unique_lesions = df.drop_duplicates(subset='lesion_id')

    train_val_l, test_l = train_test_split(unique_lesions['lesion_id'], test_size=0.15, random_state=16, stratify=unique_lesions['dx'])
    train_val_df = df[df['lesion_id'].isin(train_val_l)]
    train_val_unique = train_val_df.drop_duplicates(subset='lesion_id')

    train_l, val_l = train_test_split(train_val_unique['lesion_id'], test_size=0.17647, random_state=30, stratify=train_val_unique['dx'])
    train_df = df[df['lesion_id'].isin(train_l)].copy()
    val_df = df[df['lesion_id'].isin(val_l)].copy()
    test_df = df[df['lesion_id'].isin(test_l)].copy()

    print_flush(f"InceptionV3 Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Untouched)")

    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_ds = HAM10000Dataset(train_df, train_transform)
    val_ds   = HAM10000Dataset(val_df, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, len(val_df)

def train_inceptionv3(epochs=2, batch_size=64, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_flush(f"Training InceptionV3 on device: {device}")

    train_loader, val_loader, val_size = get_data_loaders(batch_size=batch_size)

    init_ckpt = MODELS_DIR / "inceptionv3_candidate.pth"
    ckpt_path = init_ckpt if init_ckpt.exists() else None
    model = get_inceptionv3_model(num_classes=NUM_CLASSES, checkpoint_path=ckpt_path, device=device, eval_mode=False)

    # Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification heads
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.AuxLogits.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_loss = float('inf')

    print_flush("=" * 70)
    print_flush("STARTING INCEPTIONV3 TRAINING (STANDARD CROSS-ENTROPY LOSS - NO CLASS WEIGHTS)")
    print_flush("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, aux_outputs = model(images)
            loss_main = criterion(outputs, labels)
            loss_aux  = criterion(aux_outputs, labels)
            loss = loss_main + 0.4 * loss_aux

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = train_correct / train_total

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc  = val_correct / val_total
        elapsed = time.time() - t0

        print_flush(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                    f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
                    f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc >= best_val_acc or epoch == epochs:
            best_val_acc = max(best_val_acc, epoch_val_acc)
            best_val_loss = epoch_val_loss
            print_flush(f"  --> Saving checkpoint: {CHECKPOINT_PATH.name} (Val Acc: {epoch_val_acc*100:.2f}%)")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": epoch_val_acc,
                "val_loss": best_val_loss,
                "class_names": CLASS_NAMES,
                "architecture": "InceptionV3 (No Class Weights)"
            }, str(CHECKPOINT_PATH))

    print_flush("=" * 70)
    print_flush(f"TRAINING COMPLETE. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print_flush(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print_flush("=" * 70)

if __name__ == "__main__":
    train_inceptionv3(epochs=4, batch_size=64, lr=3e-4)
