"""
End-to-End Verification Test Script for DermaAI Web Application.
Tests:
1. Home page reachability
2. Signup user registration
3. User login & session creation
4. Dashboard access
5. Image Upload & EfficientNet-B0 + Grad-CAM prediction endpoint
6. Database insertion into PREDICTIONS_HISTORY / SQLite fallback
7. Scan history retrieval
8. User logout
"""

import sys
import time
import subprocess
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:5000"
SAMPLE_IMG = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_images_part_1" / "ISIC_0024306.jpg"

def is_server_running():
    try:
        r = requests.get(f"{BASE_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def run_e2e_test():
    server_process = None
    if not is_server_running():
        print("Flask server is not currently running. Starting background app.py server...")
        server_process = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "app.py")],
            cwd=str(PROJECT_ROOT)
        )
        # Poll until server starts up (up to 15 seconds)
        started = False
        for _ in range(30):
            time.sleep(0.5)
            if is_server_running():
                started = True
                break
        if not started:
            if server_process:
                server_process.terminate()
            raise RuntimeError("Timed out waiting for Flask app.py server to start.")
        print("Flask server started successfully.")

    try:
        session = requests.Session()
        print("=" * 60)
        print("DERMA-AI FULL-STACK E2E VERIFICATION TEST")
        print("=" * 60)

        # 1. Test Home Page
        r = session.get(f"{BASE_URL}/")
        assert r.status_code == 200, f"Home page failed: {r.status_code}"
        print("[PASS] 1. Home Page reachable (HTTP 200)")

        # 2. Test Signup
        test_email = f"e2e_user_{int(time.time())}@dermaai.org"
        signup_data = {
            "name": "Dr. E2E Tester",
            "email": test_email,
            "password": "Password123!"
        }
        r = session.post(f"{BASE_URL}/signup", data=signup_data, allow_redirects=True)
        assert r.status_code == 200, f"Signup failed: {r.status_code}"
        assert "Account created" in r.text or "login" in r.url, "Signup response did not indicate success"
        print(f"[PASS] 2. Signup successful for user: {test_email}")

        # 3. Test Login
        login_data = {
            "email": test_email,
            "password": "Password123!"
        }
        r = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=True)
        assert r.status_code == 200, f"Login failed: {r.status_code}"
        assert "Dashboard" in r.text or "Welcome" in r.text, "Login redirect did not show Dashboard"
        print("[PASS] 3. Login successful & Session authenticated")

        # 4. Test Dashboard Access
        r = session.get(f"{BASE_URL}/dashboard")
        assert r.status_code == 200, f"Dashboard failed: {r.status_code}"
        print("[PASS] 4. Dashboard accessible")

        # 5. Test Image Prediction API with Grad-CAM
        assert SAMPLE_IMG.exists(), f"Sample image missing: {SAMPLE_IMG}"
        with open(SAMPLE_IMG, "rb") as img_file:
            files = {"file": ("ISIC_0024306.jpg", img_file, "image/jpeg")}
            r = session.post(f"{BASE_URL}/api/predict", files=files)

        assert r.status_code == 200, f"API Predict failed: {r.status_code} - {r.text}"
        res_json = r.json()
        assert res_json["success"] is True, f"Prediction unsuccessful: {res_json}"
        assert "predicted_class" in res_json, "Missing predicted_class in JSON"
        assert "gradcam_image_b64" in res_json, "Missing gradcam_image_b64 in JSON"
        assert "clinical_assessment" in res_json, "Missing clinical_assessment in JSON"

        pred_cls = res_json["predicted_class"]
        conf = res_json["confidence"]
        risk = res_json["clinical_assessment"]["risk_level"]
        print(f"[PASS] 5. AI Prediction API executed successfully:")
        print(f"       -> Predicted Disease : {pred_cls.upper()}")
        print(f"       -> Confidence        : {conf * 100:.2f}%")
        print(f"       -> Clinical Risk Tier: {risk}")
        print(f"       -> Grad-CAM Base64   : Received ({len(res_json['gradcam_image_b64'])} chars)")

        # 6. Test Scan History Page (verifying record saved to DB)
        r = session.get(f"{BASE_URL}/history")
        assert r.status_code == 200, f"History page failed: {r.status_code}"
        expected_conf_str = f"{conf * 100:.2f}%"
        assert pred_cls.upper() in r.text, f"Prediction diagnosis {pred_cls.upper()} not found in history HTML"
        assert expected_conf_str in r.text, f"Confidence {expected_conf_str} not found in history HTML"
        print(f"[PASS] 6. Database record persisted & verified in History view ({pred_cls.upper()}, {expected_conf_str})")

        # 7. Test Logout
        r = session.get(f"{BASE_URL}/logout", allow_redirects=True)
        assert r.status_code == 200, f"Logout failed: {r.status_code}"
        print("[PASS] 7. Logout successful")

        print("=" * 60)
        print("ALL 7 END-TO-END VERIFICATION CHECKS PASSED PERFECTLY!")
        print("=" * 60)
    finally:
        if server_process:
            print("Shutting down background Flask server process...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    run_e2e_test()
