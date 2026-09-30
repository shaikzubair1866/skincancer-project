"""
Component & Integration Verification Test for 3-Model Ensemble System.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import model
from models.efficientnet_b0_model import get_efficientnet_b0_model, get_efficientnet_b0_transform, predict_efficientnet_b0
from models.resnet50_model import get_resnet50_model, get_resnet50_transform, predict_resnet50
from models.inceptionv3_model import get_inceptionv3_model, get_inceptionv3_transform, predict_inceptionv3

SAMPLE_IMG_PATH = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1" / "ISIC_0024306.jpg"

def run_tests():
    print("=" * 70)
    print("TESTING 3-MODEL ENSEMBLE COMPONENTS & INFERENCE")
    print("=" * 70)

    # 1. Test Image Presence
    assert SAMPLE_IMG_PATH.exists(), f"Sample image not found: {SAMPLE_IMG_PATH}"
    img = Image.open(SAMPLE_IMG_PATH).convert("RGB")
    print(f"[PASS] Sample image loaded successfully: {SAMPLE_IMG_PATH.name} ({img.size})")

    # 2. Test Checkpoint Files
    ckpt_b0 = (PROJECT_ROOT / "models" / "efficientnet_b0_phase3_best.pth") if (PROJECT_ROOT / "models" / "efficientnet_b0_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "efficientnet_b0_phase3_best.pth")
    ckpt_res = (PROJECT_ROOT / "models" / "resnet50_phase3_best.pth") if (PROJECT_ROOT / "models" / "resnet50_phase3_best.pth").exists() else (PROJECT_ROOT / "backups" / "79_51_baseline" / "resnet50_phase3_best.pth")
    ckpt_inc = PROJECT_ROOT / "models" / "inceptionv3_candidate.pth"

    assert ckpt_b0.exists(), f"Missing EfficientNet-B0 checkpoint: {ckpt_b0}"
    assert ckpt_res.exists(), f"Missing ResNet-50 checkpoint: {ckpt_res}"
    assert ckpt_inc.exists(), f"Missing InceptionV3 checkpoint: {ckpt_inc}"
    print("[PASS] All 3 checkpoint files verified on disk.")

    # 3. Test Model 1: EfficientNet-B0
    m_b0 = get_efficientnet_b0_model(num_classes=7, checkpoint_path=ckpt_b0, device="cpu", eval_mode=True)
    tf_b0 = get_efficientnet_b0_transform()
    t_b0 = tf_b0(img).unsqueeze(0)
    assert t_b0.shape == (1, 3, 256, 256), f"Unexpected shape for B0: {t_b0.shape}"
    p_b0 = predict_efficientnet_b0(m_b0, t_b0, device="cpu")
    assert len(p_b0) == 7, f"Expected 7 classes, got {len(p_b0)}"
    assert np.isclose(np.sum(p_b0), 1.0, atol=1e-3), f"Probabilities do not sum to 1: {np.sum(p_b0)}"
    print(f"[PASS] EfficientNet-B0 inference successful (Top class: {np.argmax(p_b0)}, Prob: {np.max(p_b0):.4f})")

    # 4. Test Model 2: ResNet-50
    m_res = get_resnet50_model(num_classes=7, checkpoint_path=ckpt_res, device="cpu", eval_mode=True)
    tf_res = get_resnet50_transform()
    t_res = tf_res(img).unsqueeze(0)
    assert t_res.shape == (1, 3, 256, 256), f"Unexpected shape for ResNet: {t_res.shape}"
    p_res = predict_resnet50(m_res, t_res, device="cpu")
    assert len(p_res) == 7, f"Expected 7 classes, got {len(p_res)}"
    assert np.isclose(np.sum(p_res), 1.0, atol=1e-3), f"Probabilities do not sum to 1: {np.sum(p_res)}"
    print(f"[PASS] ResNet-50 inference successful (Top class: {np.argmax(p_res)}, Prob: {np.max(p_res):.4f})")

    # 5. Test Model 3: InceptionV3
    m_inc = get_inceptionv3_model(num_classes=7, checkpoint_path=ckpt_inc, device="cpu", eval_mode=True)
    tf_inc = get_inceptionv3_transform()
    t_inc = tf_inc(img).unsqueeze(0)
    assert t_inc.shape == (1, 3, 299, 299), f"Unexpected shape for InceptionV3: {t_inc.shape}"
    p_inc = predict_inceptionv3(m_inc, t_inc, device="cpu")
    assert len(p_inc) == 7, f"Expected 7 classes, got {len(p_inc)}"
    assert np.isclose(np.sum(p_inc), 1.0, atol=1e-3), f"Probabilities do not sum to 1: {np.sum(p_inc)}"
    print(f"[PASS] InceptionV3 inference successful (Top class: {np.argmax(p_inc)}, Prob: {np.max(p_inc):.4f})")

    # 6. Test Production 3-Model Ensemble (model.py)
    res = model.predict(img, generate_cam=True)
    assert res["predicted_class"] in model.DEFAULT_CLASS_NAMES, f"Invalid predicted class: {res['predicted_class']}"
    assert 0.0 <= res["confidence"] <= 1.0, f"Invalid confidence: {res['confidence']}"
    assert len(res["probabilities"]) == 7, f"Expected 7 class probabilities, got {len(res['probabilities'])}"
    assert "gradcam_image" in res, "Missing gradcam_image in prediction results"
    print(f"[PASS] Production 3-Model Ensemble predict() execution successful:")
    print(f"       -> Class      : {res['predicted_class'].upper()}")
    print(f"       -> Confidence : {res['confidence'] * 100:.2f}%")
    print(f"       -> Grad-CAM   : Generated image size {res['gradcam_image'].size}")

    print("=" * 70)
    print("ALL COMPONENT & ENSEMBLE TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
