"""
Skin Cancer Detection Web Application (Finalized Production)
Flask application integrating 3-Model Ensemble (EfficientNet-B0 + ResNet-50 + InceptionV3) + 5-View TTA inference,
Grad-CAM heatmap visualization, Oracle database auth & prediction history, and clinical risk evaluation.
Verified Test Accuracy: 91.17%
"""

import os
import uuid
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_from_directory
)

import auth
import database
import model

# ─── Flask App Configuration ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
app.secret_key = os.getenv("SECRET_KEY", "skincancer_secret_key_2026_secure")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

# Initialize database tables on application startup
with app.app_context():
    database.init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_authenticated():
    return 'user' in session and session['user'] is not None

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    user = session.get('user')
    return render_template('index.html', user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if is_authenticated():
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        res = auth.signup(name, email, password)
        if res['success']:
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login_page'))
        else:
            flash(res['message'], "danger")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if is_authenticated():
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        res = auth.login(email, password)
        if res['success']:
            session['user'] = res['user']
            session['token'] = res['token']
            flash(f"Welcome back, {res['user']['name']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash(res['message'], "danger")

    return render_template('login.html')

@app.route('/logout')
def logout_page():
    session.clear()
    flash("You have been logged out safely.", "info")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if not is_authenticated():
        flash("Please log in to access your dashboard.", "warning")
        return redirect(url_for('login_page'))

    user = session['user']
    history = database.get_user_prediction_history(user['user_id'])
    return render_template('dashboard.html', user=user, history=history[:5], total_scans=len(history))

@app.route('/predict', methods=['GET'])
def predict_page():
    if not is_authenticated():
        flash("Please log in to run skin lesion predictions.", "warning")
        return redirect(url_for('login_page'))

    return render_template('predict.html', user=session['user'])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not is_authenticated():
        return jsonify({"success": False, "message": "Authentication required."}), 401

    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type. Only JPG, PNG, and WEBP images are allowed."}), 400

    try:
        image_bytes = file.read()
        # Validate that bytes represent a real image
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.verify()
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "message": "Uploaded file is corrupted or not a valid image."}), 400

    # Save original image file locally
    filename_stem = f"{uuid.uuid4().hex}"
    orig_filename = f"{filename_stem}_orig.jpg"
    cam_filename = f"{filename_stem}_gradcam.jpg"

    orig_path = UPLOAD_FOLDER / orig_filename
    cam_path = UPLOAD_FOLDER / cam_filename

    pil_img.save(orig_path, "JPEG")

    print("========== UPLOAD DEBUG ==========")
    print("Filename:", file.filename)
    print("Image size:", pil_img.size)
    print("Image mode:", pil_img.mode)

    import numpy as np
    arr = np.array(pil_img)

    print("Pixel min:", arr.min())
    print("Pixel max:", arr.max())
    print("Pixel mean:", round(float(arr.mean()), 4))
    print("Pixel std:", round(float(arr.std()), 4))
    print("==================================")

    # Perform Inference with Grad-CAM and Risk Engine
    try:
        results = model.predict(pil_img, generate_cam=True)
    except Exception as e:
        return jsonify({"success": False, "message": f"Inference engine error: {str(e)}"}), 500

    print("===== PREDICTION DEBUG =====")
    print("Uploaded Filename:", file.filename)
    print("PIL Image size:   ", pil_img.size)
    print("Predicted:        ", results.get("predicted_class"))
    print("Confidence:       ", results.get("confidence"))
    print("Probabilities:    ", results.get("probabilities"))
    print("============================")

    # Save Grad-CAM image overlay
    gradcam_pil = results["gradcam_image"]
    gradcam_pil.save(cam_path, "JPEG")

    pred_class = results["predicted_class"]
    confidence = results["confidence"]
    clinical_assessment = results["clinical_assessment"]

    # Save record to Database
    user = session['user']
    database.insert_prediction(
        user_id=user['user_id'],
        image_reference=orig_filename,
        predicted_class=pred_class,
        confidence=confidence
    )

    # Convert Grad-CAM overlay image to Base64 for instantaneous UI render
    buffered = BytesIO()
    gradcam_pil.save(buffered, format="JPEG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Convert Original image to Base64
    buf_orig = BytesIO()
    pil_img.save(buf_orig, format="JPEG")
    orig_b64 = base64.b64encode(buf_orig.getvalue()).decode('utf-8')

    return jsonify({
        "success": True,
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": results["probabilities"],
        "clinical_assessment": clinical_assessment,
        "orig_image_b64": f"data:image/jpeg;base64,{orig_b64}",
        "gradcam_image_b64": f"data:image/jpeg;base64,{gradcam_b64}",
        "orig_image_url": f"/static/uploads/{orig_filename}",
        "gradcam_image_url": f"/static/uploads/{cam_filename}"
    })

@app.route('/history')
def history_page():
    if not is_authenticated():
        flash("Please log in to view scan history.", "warning")
        return redirect(url_for('login_page'))

    user = session['user']
    history = database.get_user_prediction_history(user['user_id'])
    
    # Enrich history with clinical risk details
    for item in history:
        assessment = model.get_clinical_assessment(item['predicted_class'], item['confidence'])
        item['category_name'] = assessment['category_name']
        item['risk_level'] = assessment['risk_level']
        item['urgency_badge'] = assessment['urgency_badge']

    return render_template('history.html', user=user, history=history)

@app.route('/about')
def about_page():
    return render_template('about.html', user=session.get('user'))

@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    if request.method == 'POST':
        flash("Thank you for contacting us! Our medical support team will respond shortly.", "success")
        return redirect(url_for('contact_page'))
    return render_template('contact.html', user=session.get('user'))

if __name__ == '__main__':
    print("Starting Skin Cancer Detection Web Application on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
