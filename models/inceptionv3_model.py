"""
InceptionV3 Architecture Definition, Preprocessing, and Inference Loader.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import numpy as np

NUM_CLASSES = 7
IMAGE_SIZE = 299
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

def get_inceptionv3_transform():
    """Returns model-specific preprocessing pipeline for InceptionV3 (299x299)."""
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

def get_inceptionv3_model(num_classes=NUM_CLASSES, checkpoint_path=None, pretrained=False, device="cpu", eval_mode=True):
    """Instantiate InceptionV3 model and optionally load checkpoint weights."""
    device = torch.device(device)
    weights = models.Inception_V3_Weights.DEFAULT if (pretrained and checkpoint_path is None) else None
    model = models.inception_v3(weights=weights)

    num_ftrs_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"InceptionV3 checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.to(device)
    if eval_mode:
        model.eval()
    else:
        model.train()

    return model

def predict_inceptionv3(model, raw_tensor, device="cpu"):
    """Run inference for InceptionV3 on a normalized 4D tensor (1, 3, 299, 299)."""
    model.eval()
    with torch.no_grad():
        outputs = model(raw_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return probs
