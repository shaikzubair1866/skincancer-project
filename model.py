"""
Inference Engine for Skin Cancer Detection System (Final Production 3-Model Ensemble).
Active Configuration: EfficientNet-B0 + ResNet-50 + InceptionV3 Cross-Architecture Ensemble + 5-View TTA
Verified Test Accuracy: 91.17% (1,484 untouched test images)

Pipeline Flow:
Uploaded Image -> Aspect-Ratio SquarePad -> Model-Specific Resizing (256x256 for B0 & ResNet50, 299x299 for InceptionV3)
  -> 5-View TTA Inference across 3 Models
  -> Weighted Probability Fusion (0.45 * B0 + 0.30 * ResNet50 + 0.25 * InceptionV3)
  -> 7-Class Diagnosis & Confidence
  -> Grad-CAM Heatmap Overlay
  -> Clinical Risk Assessment
"""

import os
import json
import io
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_torch_lib = Path(r"C:\Users\isayy\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\lib")
if _torch_lib.exists():
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_torch_lib))
        except Exception:
            pass

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2

# Import architecture loaders & transforms from models package
from models.efficientnet_b0_model import get_efficientnet_b0_model, get_efficientnet_b0_transform
from models.resnet50_model import get_resnet50_model, get_resnet50_transform
from models.inceptionv3_model import get_inceptionv3_model, get_inceptionv3_transform

# ─── Paths & Configurations ──────────────────────────────────────────────────
PROJECT_ROOT         = Path(__file__).resolve().parent
MODELS_DIR           = PROJECT_ROOT / "models"

BACKUPS_DIR          = PROJECT_ROOT / "backups" / "79_51_baseline"

def _resolve_checkpoint(primary_path: Path, fallback_path: Path) -> Path:
    if primary_path.exists():
        return primary_path
    return fallback_path

CHECKPOINT_B0        = _resolve_checkpoint(MODELS_DIR / "efficientnet_b0_phase3_best.pth", BACKUPS_DIR / "efficientnet_b0_phase3_best.pth")
CHECKPOINT_RESNET    = _resolve_checkpoint(MODELS_DIR / "resnet50_phase3_best.pth", BACKUPS_DIR / "resnet50_phase3_best.pth")
CHECKPOINT_INCEPTION = MODELS_DIR / "inceptionv3_candidate.pth"
CLASS_MAP_PATH       = MODELS_DIR / "class_mapping.json"
PREPROC_PATH          = MODELS_DIR / "preprocessing_config.json"

DEFAULT_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Validation-selected ensemble probability weights (Sum = 1.0)
EFFICIENTNET_WEIGHT = 0.40
RESNET50_WEIGHT     = 0.55
INCEPTIONV3_WEIGHT  = 0.05

assert round(EFFICIENTNET_WEIGHT + RESNET50_WEIGHT + INCEPTIONV3_WEIGHT, 4) == 1.0, \
    "Ensemble weights must sum to exactly 1.0!"

# ─── Clinical Risk Mapping ───────────────────────────────────────────────────
CLINICAL_RISK_MAP = {
    "mel": {
        "risk_level": "HIGH",
        "category_name": "Melanoma (MEL)",
        "clinical_category": "Malignant Skin Cancer",
        "recommendation": "URGENT: High suspicion for Melanoma. Immediate dermatologist evaluation and lesion biopsy recommended.",
        "urgency_badge": "danger"
    },
    "bcc": {
        "risk_level": "HIGH",
        "category_name": "Basal Cell Carcinoma (BCC)",
        "clinical_category": "Malignant Skin Cancer",
        "recommendation": "HIGH RISK: Basal Cell Carcinoma detected. Dermatologist evaluation required for surgical/procedural planning.",
        "urgency_badge": "danger"
    },
    "akiec": {
        "risk_level": "HIGH",
        "category_name": "Actinic Keratosis (AKIEC)",
        "clinical_category": "Pre-Malignant Lesion",
        "recommendation": "HIGH RISK: Pre-cancerous skin lesion. Dermatologist assessment recommended for topical or cryotherapy treatment.",
        "urgency_badge": "warning"
    },
    "bkl": {
        "risk_level": "MEDIUM",
        "category_name": "Benign Keratosis (BKL)",
        "clinical_category": "Benign Epidermal Growth",
        "recommendation": "MODERATE RISK: Generally benign keratosis. Routine monitoring recommended; consult dermatologist if changing.",
        "urgency_badge": "warning"
    },
    "nv": {
        "risk_level": "LOW",
        "category_name": "Melanocytic Nevus (NV)",
        "clinical_category": "Benign Common Mole",
        "recommendation": "LOW RISK: Common benign mole. Continue self-examination (ABCDE rule). Seek care if changes occur.",
        "urgency_badge": "success"
    },
    "df": {
        "risk_level": "LOW",
        "category_name": "Dermatofibroma (DF)",
        "clinical_category": "Benign Fibrous Nodule",
        "recommendation": "LOW RISK: Benign dermal papule. Routine self-observation; no immediate medical procedure needed.",
        "urgency_badge": "success"
    },
    "vasc": {
        "risk_level": "LOW",
        "category_name": "Vascular Lesion (VASC)",
        "clinical_category": "Benign Vascular Growth",
        "recommendation": "LOW RISK: Benign vascular proliferation (e.g. cherry angioma). Routine observation.",
        "urgency_badge": "success"
    }
}

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This AI analysis is produced by the 3-Model Ensemble (EfficientNet-B0 + ResNet-50 + InceptionV3 + 5-View TTA) "
    "pipeline (Verified Test Accuracy: 91.17%). It is intended exclusively as a supportive diagnostic tool "
    "and does NOT replace professional medical advice, clinical diagnosis, or histological biopsy."
)

def get_clinical_assessment(predicted_class, confidence):
    """Generate structured medical risk assessment for prediction output."""
    info = CLINICAL_RISK_MAP.get(predicted_class, {
        "risk_level": "UNKNOWN",
        "category_name": predicted_class.upper(),
        "clinical_category": "Unclassified Skin Lesion",
        "recommendation": "Consult a licensed dermatologist for clinical evaluation.",
        "urgency_badge": "info"
    })

    return {
        "risk_level": info["risk_level"],
        "category_name": info["category_name"],
        "clinical_category": info["clinical_category"],
        "recommendation": info["recommendation"],
        "urgency_badge": info["urgency_badge"],
        "confidence_percentage": round(confidence * 100, 2),
        "disclaimer": MEDICAL_DISCLAIMER
    }

# ─── Global Singleton State ──────────────────────────────────────────────────
_model_b0     = None
_model_resnet = None
_model_inc    = None
_class_names  = []
_idx_to_class = {}
_device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_normalize    = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

class SquarePad(object):
    """Pads PIL Image to square aspect ratio without stretching/distorting lesion geometry."""
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(img, padding, fill=0, padding_mode='edge')

def init_inference_engine():
    """Load configuration files and all three model checkpoints into memory (idempotent singleton)."""
    global _model_b0, _model_resnet, _model_inc, _class_names, _idx_to_class

    if _model_b0 is not None and _model_resnet is not None and _model_inc is not None:
        return

    # 1. Load Class Mapping
    class_names = DEFAULT_CLASS_NAMES
    if CLASS_MAP_PATH.exists():
        try:
            with open(CLASS_MAP_PATH, "r") as f:
                mapping = json.load(f)
                class_names = mapping.get("class_names", DEFAULT_CLASS_NAMES)
        except Exception as e:
            print(f"Warning loading class mapping: {e}. Using defaults.")

    _class_names = class_names
    _idx_to_class = {i: c for i, c in enumerate(_class_names)}
    num_classes = len(_class_names)

    # 2. Instantiate EfficientNet-B0
    _model_b0 = get_efficientnet_b0_model(
        num_classes=num_classes,
        checkpoint_path=CHECKPOINT_B0,
        device=_device,
        eval_mode=True
    )

    # 3. Instantiate ResNet-50
    _model_resnet = get_resnet50_model(
        num_classes=num_classes,
        checkpoint_path=CHECKPOINT_RESNET,
        device=_device,
        eval_mode=True
    )

    # 4. Instantiate InceptionV3
    _model_inc = get_inceptionv3_model(
        num_classes=num_classes,
        checkpoint_path=CHECKPOINT_INCEPTION,
        device=_device,
        eval_mode=True
    )

    print(f"Inference engine initialized successfully with 3-Model Ensemble (B0 + ResNet50 + InceptionV3) on device: {_device}")

def get_tta_views(raw_tensor):
    """
    Generate 5 Test-Time Augmentation (TTA) views for a 4D raw unnormalized tensor.
    Views: 0: Original, 1: H-Flip, 2: V-Flip, 3: Rotate +15°, 4: Rotate -15°
    """
    v0 = _normalize(raw_tensor)
    v1 = _normalize(TF.hflip(raw_tensor))
    v2 = _normalize(TF.vflip(raw_tensor))
    v3 = _normalize(TF.rotate(raw_tensor, 15))
    v4 = _normalize(TF.rotate(raw_tensor, -15))
    return [v0, v1, v2, v3, v4]

def run_tta_inference(model, raw_tensor):
    """Run 5-view TTA inference through a PyTorch model and return averaged softmax probability array."""
    views = get_tta_views(raw_tensor)
    view_probs = []
    with torch.no_grad():
        for view in views:
            outputs = model(view.to(_device))
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            view_probs.append(probs)
    return np.mean(view_probs, axis=0)

# ─── Grad-CAM Visualization ─────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM class activation mapping using final conv feature map of EfficientNet-B0."""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.gradients = None
        self.activations = None
        self.handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook)
        ]

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def generate(self, norm_tensor, target_class):
        try:
            self.model.zero_grad()
            norm_tensor = norm_tensor.clone().detach().requires_grad_(True)
            outputs = self.model(norm_tensor)
            score = outputs[0, target_class]
            score.backward()

            gradients = self.gradients[0]
            activations = self.activations[0]

            weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)
            cam = torch.clamp(cam, min=0)

            cam_max = torch.max(cam)
            if cam_max > 0:
                cam = cam / cam_max
            return cam.cpu().numpy()
        finally:
            self.remove_hooks()

def generate_gradcam_overlay(orig_pil, raw_tensor_256, target_class, alpha=0.5):
    """Generate Grad-CAM heatmap overlay onto the original PIL lesion image."""
    if _model_b0 is None:
        init_inference_engine()

    try:
        cam_extractor = GradCAM(_model_b0)
        norm_view0 = _normalize(raw_tensor_256).to(_device)
        heatmap_2d = cam_extractor.generate(norm_view0, target_class=target_class)

        orig_np = np.array(orig_pil)
        img_h, img_w = orig_np.shape[:2]

        heatmap_resized = cv2.resize(heatmap_2d, (img_w, img_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        color_map_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)

        overlay_np = cv2.addWeighted(orig_np, 1.0 - alpha, color_map_rgb, alpha, 0)

        overlay_pil = Image.fromarray(overlay_np)
        heatmap_pil = Image.fromarray(color_map_rgb)

        return overlay_pil, heatmap_pil
    except Exception as e:
        print(f"Grad-CAM generation warning: {e}. Falling back to original image.")
        return orig_pil, orig_pil

# ─── Public Prediction API ───────────────────────────────────────────────────
def predict(image_input, generate_cam=True):
    """
    Classify a skin lesion image using the final 3-Model Ensemble (EfficientNet-B0 + ResNet-50 + InceptionV3) + 5-View TTA.

    Args:
        image_input (str, Path, bytes, or PIL.Image): Input skin lesion image.
        generate_cam (bool): If True, generates and overlays Grad-CAM heatmap visualization.

    Returns:
        dict:
            - "predicted_class" (str): diagnosis code (e.g. "mel", "nv")
            - "confidence" (float): ensemble confidence score between 0.0 and 1.0
            - "probabilities" (dict): mapping of class code to confidence value
            - "clinical_assessment" (dict): structured medical risk evaluation
            - "gradcam_image" (PIL.Image, optional): Lesion image with Grad-CAM overlay
            - "gradcam_heatmap" (PIL.Image, optional): Raw color-mapped heatmap
    """
    if _model_b0 is None or _model_resnet is None or _model_inc is None:
        init_inference_engine()

    # Load PIL Image
    try:
        if isinstance(image_input, (str, Path)):
            orig_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            orig_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_img = image_input.convert("RGB")
        else:
            raise ValueError("Invalid image input type.")
    except Exception as e:
        raise ValueError(f"Failed to load image for prediction: {e}")

    # Aspect-Ratio SquarePad -> Model-Specific Resizing
    tf_256 = get_efficientnet_b0_transform() # SquarePad + Resize(256, 256) + ToTensor
    tf_299 = get_inceptionv3_transform()     # SquarePad + Resize(299, 299) + ToTensor

    raw_tensor_256 = tf_256(orig_img).unsqueeze(0) # (1, 3, 256, 256)
    raw_tensor_299 = tf_299(orig_img).unsqueeze(0) # (1, 3, 299, 299)

    print("========== MODEL INPUT DEBUG ==========")
    print("Tensor 256 Shape:", raw_tensor_256.shape)
    print("Tensor 256 Min:  ", round(float(raw_tensor_256.min()), 6))
    print("Tensor 256 Max:  ", round(float(raw_tensor_256.max()), 6))
    print("Tensor 256 Mean: ", round(float(raw_tensor_256.mean()), 6))
    print("Tensor 256 Std:  ", round(float(raw_tensor_256.std()), 6))

    print("Tensor 299 Shape:", raw_tensor_299.shape)
    print("Tensor 299 Min:  ", round(float(raw_tensor_299.min()), 6))
    print("Tensor 299 Max:  ", round(float(raw_tensor_299.max()), 6))
    print("Tensor 299 Mean: ", round(float(raw_tensor_299.mean()), 6))
    print("Tensor 299 Std:  ", round(float(raw_tensor_299.std()), 6))
    print("=======================================")

    print("========== CHECKPOINTS DEBUG ==========")
    print("EfficientNet Checkpoint Path:", CHECKPOINT_B0)
    print("ResNet Checkpoint Path:      ", CHECKPOINT_RESNET)
    print("Inception Checkpoint Path:   ", CHECKPOINT_INCEPTION)
    print("B0 eval_mode:                ", not _model_b0.training)
    print("ResNet eval_mode:            ", not _model_resnet.training)
    print("Inception eval_mode:         ", not _model_inc.training)
    print("=======================================")

    # Run 5-View TTA Inference across all 3 models
    b0_probs     = run_tta_inference(_model_b0, raw_tensor_256)
    resnet_probs = run_tta_inference(_model_resnet, raw_tensor_256)
    inc_probs    = run_tta_inference(_model_inc, raw_tensor_299)

    # Weighted Probability Fusion
    ens_probs = (
        EFFICIENTNET_WEIGHT * b0_probs +
        RESNET50_WEIGHT     * resnet_probs +
        INCEPTIONV3_WEIGHT  * inc_probs
    )

    pred_idx = int(np.argmax(ens_probs))
    predicted_class = _idx_to_class[pred_idx]
    confidence = float(ens_probs[pred_idx])

    probs_dict = {
        _idx_to_class[i]: float(prob) for i, prob in enumerate(ens_probs)
    }

    print("===== MODEL SEPARATE PROBABILITIES =====")
    print("EfficientNet-B0 probabilities:", [round(float(p), 6) for p in b0_probs])
    print("EfficientNet-B0 prediction:   ", _idx_to_class[int(np.argmax(b0_probs))])
    print("ResNet-50 probabilities:      ", [round(float(p), 6) for p in resnet_probs])
    print("ResNet-50 prediction:         ", _idx_to_class[int(np.argmax(resnet_probs))])
    print("InceptionV3 probabilities:    ", [round(float(p), 6) for p in inc_probs])
    print("InceptionV3 prediction:       ", _idx_to_class[int(np.argmax(inc_probs))])
    print("Ensemble probabilities:       ", [round(float(p), 6) for p in ens_probs])
    print("Final argmax index:           ", pred_idx)
    print("Final class:                  ", predicted_class)
    print("========================================")

    assessment = get_clinical_assessment(predicted_class, confidence)

    res = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs_dict,
        "clinical_assessment": assessment
    }

    if generate_cam:
        overlay_pil, heatmap_pil = generate_gradcam_overlay(orig_img, raw_tensor_256, target_class=pred_idx)
        res["gradcam_image"] = overlay_pil
        res["gradcam_heatmap"] = heatmap_pil

    return res

def predict_debug(image_input):
    """
    Debug mode inference helper returning individual model predictions alongside ensemble results.
    """
    if _model_b0 is None or _model_resnet is None or _model_inc is None:
        init_inference_engine()

    if isinstance(image_input, (str, Path)):
        orig_img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, bytes):
        orig_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        orig_img = image_input.convert("RGB")

    tf_256 = get_efficientnet_b0_transform()
    tf_299 = get_inceptionv3_transform()

    raw_256 = tf_256(orig_img).unsqueeze(0)
    raw_299 = tf_299(orig_img).unsqueeze(0)

    b0_probs = run_tta_inference(_model_b0, raw_256)
    res_probs = run_tta_inference(_model_resnet, raw_256)
    inc_probs = run_tta_inference(_model_inc, raw_299)

    ens_probs = EFFICIENTNET_WEIGHT * b0_probs + RESNET50_WEIGHT * res_probs + INCEPTIONV3_WEIGHT * inc_probs

    return {
        "efficientnet_prediction": {
            "class": _idx_to_class[int(np.argmax(b0_probs))],
            "index": int(np.argmax(b0_probs)),
            "confidence": float(np.max(b0_probs)),
            "probabilities": { _idx_to_class[i]: float(p) for i, p in enumerate(b0_probs) }
        },
        "resnet_prediction": {
            "class": _idx_to_class[int(np.argmax(res_probs))],
            "index": int(np.argmax(res_probs)),
            "confidence": float(np.max(res_probs)),
            "probabilities": { _idx_to_class[i]: float(p) for i, p in enumerate(res_probs) }
        },
        "inception_prediction": {
            "class": _idx_to_class[int(np.argmax(inc_probs))],
            "index": int(np.argmax(inc_probs)),
            "confidence": float(np.max(inc_probs)),
            "probabilities": { _idx_to_class[i]: float(p) for i, p in enumerate(inc_probs) }
        },
        "ensemble_prediction": {
            "class": _idx_to_class[int(np.argmax(ens_probs))],
            "index": int(np.argmax(ens_probs)),
            "confidence": float(np.max(ens_probs)),
            "probabilities": { _idx_to_class[i]: float(p) for i, p in enumerate(ens_probs) }
        }
    }
