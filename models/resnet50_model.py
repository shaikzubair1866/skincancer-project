"""
ResNet-50 Architecture Definition, Preprocessing, and Inference Loader.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import numpy as np

NUM_CLASSES = 7
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class SquarePad(object):
    """Pads PIL Image to square aspect ratio without distorting lesion geometry."""
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(img, padding, fill=0, padding_mode='edge')

def get_resnet50_transform():
    """Returns model-specific preprocessing pipeline for ResNet-50 (256x256)."""
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

def get_resnet50_model(num_classes=NUM_CLASSES, checkpoint_path=None, device="cpu", eval_mode=True):
    """Instantiate ResNet-50 model and optionally load checkpoint weights."""
    device = torch.device(device)
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ResNet-50 checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.to(device)
    if eval_mode:
        model.eval()
    else:
        model.train()

    return model

def predict_resnet50(model, raw_tensor, device="cpu"):
    """Run inference for ResNet-50 on a normalized 4D tensor (1, 3, 256, 256)."""
    model.eval()
    with torch.no_grad():
        outputs = model(raw_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return probs
