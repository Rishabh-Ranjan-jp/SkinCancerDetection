from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models import db, Prediction
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import os
from datetime import datetime
import io
import base64

main_bp = Blueprint('main', __name__)

# ============================================
# CONFIGURATION
# ============================================
# Class mapping for multi-class model (9-class)
CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]
NUM_CLASSES = len(CLASS_NAMES)

# Binary model for benign/malignant screening
BINARY_MODEL_PATH = 'models/best_bin_model_fold1.pt'
# Multi-class model for detailed classification
MULTICLASS_MODEL_PATH = 'models/best_model_fold3.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation (224x224 for both models)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Global model variables
BINARY_MODEL = None
MULTICLASS_MODEL = None

# ============================================
# MODEL LOADING
# ============================================
def load_binary_model():
    """Load binary classifier (benign/malignant)"""
    global BINARY_MODEL
    if BINARY_MODEL is not None:
        return BINARY_MODEL
    try:
        print(f"Loading binary model from {BINARY_MODEL_PATH}...")
        # Create base EfficientNet-B0 model
        model = efficientnet_b0(weights=None)
        # Replace classifier for binary classification (2 classes)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
        # Load trained weights
        model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        BINARY_MODEL = model
        print("✓ Binary model loaded successfully!")
        return BINARY_MODEL
    except FileNotFoundError:
        print(f"❌ Binary model file not found at {BINARY_MODEL_PATH}")
        raise
    except Exception as e:
        print(f"❌ Error loading binary model: {e}")
        raise

def load_multiclass_model():
    """Load multi-class classifier (9-class)"""
    global MULTICLASS_MODEL
    if MULTICLASS_MODEL is not None:
        return MULTICLASS_MODEL
    try:
        print(f"Loading multi-class model from {MULTICLASS_MODEL_PATH}...")
        # Create base EfficientNet-B0 model
        model = efficientnet_b0(weights=None)
        # Replace classifier for 9 classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
        # Load trained weights
        model.load_state_dict(torch.load(MULTICLASS_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        MULTICLASS_MODEL = model
        print("✓ Multi-class model loaded successfully!")
        return MULTICLASS_MODEL
    except FileNotFoundError:
        print(f"❌ Multi-class model file not found at {MULTICLASS_MODEL_PATH}")
        raise
    except Exception as e:
        print(f"❌ Error loading multi-class model: {e}")
        raise

def get_risk_level(multiclass_confidence):
    """Determine risk level based on multiclass confidence"""
    if multiclass_confidence < 0.30:
        return "LOW RISK", "🟢 Low confidence - likely benign"
    elif multiclass_confidence < 0.60:
        return "MEDIUM RISK", "🟡 Medium confidence - monitor closely"
    else:
        return "HIGH RISK", "🔴 High confidence - urgent attention needed"

def preprocess_image(image_path):
    """Preprocess image for model inference"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = INFERENCE_TRANSFORM(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def preprocess_image_from_base64(base64_string):
    """Preprocess image from base64 string (from camera)"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_tensor = INFERENCE_TRANSFORM(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error preprocessing base64 image: {e}")
        raise

def predict_binary(image_path):
    """
    Stage 1: Binary classification (benign/malignant)
    Returns:
        is_malignant (bool): True if malignant, False if benign
        confidence (float): Confidence score (0-1)
    """
    model = load_binary_model()
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probabilities, 1)
        is_malignant = class_idx.item() == 1
        confidence = confidence.item()
    return is_malignant, confidence

def predict_multiclass(image_path):
    """
    Stage 2: Multi-class classification (9 skin conditions)
    Returns:
        class_name (str): Predicted skin condition
        confidence (float): Confidence score (0-1)
        all_confidences (dict): All class confidences
    """
    model = load_multiclass_model()
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probabilities, 1)
        confidence = confidence.item()
        class_idx = class_idx.item()
        class_name = CLASS_NAMES[class_idx]
        all_probs = probabilities[0].cpu().numpy()
        all_confidences = {
            CLASS_NAMES[i]: float(all_probs[i])
            for i in range(NUM_CLASSES)
        }
    return class_name, confidence, all_confidences

def predict_binary_from_base64(base64_string):
    """Binary classification from base64 image"""
    model = load_binary_model()
    img_tensor = preprocess_image_from_base64(base64_string)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probabilities, 1)
        is_malignant = class_idx.item() == 1
        confidence = confidence.item()
    return is_malignant, confidence

def predict_multiclass_from_base64(base64_string):
    """Multi-class classification from base64 image"""
    model = load_multiclass_model()
    img_tensor = preprocess_image_from_base64(base64_string)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probabilities, 1)
        confidence = confidence.item()
        class_idx = class_idx.item()
        class_name = CLASS_NAMES[class_idx]
        all_probs = probabilities[0].cpu().numpy()
        all_confidences = {
            CLASS_NAMES[i]: float(all_probs[i])
            for i in range(NUM_CLASSES)
        }
    return class_name, confidence, all_confidences

# ============================================
# FLASK ROUTES
# ============================================
@main_bp.route('/')
def index():
    """Home page"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Two-stage ensemble prediction:
    Stage 1: Binary (benign/malignant)
    Stage 2: Multi-class (if malignant)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        # Save image
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = secure_filename(f"{timestamp}{file.filename}")
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        print(f"Making ensemble prediction for: {filename}")

        # STAGE 1: Binary classification
        is_malignant, binary_confidence = predict_binary(filepath)
        result = {
            'success': True,
            'stage1': {
                'is_malignant': is_malignant,
                'binary_label': 'Malignant' if is_malignant else 'Benign',
                'binary_confidence': float(binary_confidence),
                'binary_confidence_percent': float(binary_confidence * 100)
            },
            'image_path': f'uploads/{filename}'
        }

        # STAGE 2: Multi-class classification (only if malignant)
        if is_malignant:
            class_name, multiclass_confidence, all_confidences = predict_multiclass(filepath)
            
            # NEW: Add risk assessment
            risk_level, risk_description = get_risk_level(multiclass_confidence)
            
            result['stage2'] = {
                'condition': class_name,
                'confidence': float(multiclass_confidence),
                'confidence_percent': float(multiclass_confidence * 100),
                'risk_level': risk_level,  # NEW
                'risk_description': risk_description,  # NEW
                'all_confidences': all_confidences
            }
            print(f"MALIGNANT ALERT: {class_name} ({multiclass_confidence*100:.2f}%) - {risk_level}")
        else:
            print(f"Benign classification confirmed")

        # Save to database
        pred_record = Prediction(
            user_id=current_user.id,
            image_path=f'uploads/{filename}',
            binary_prediction='malignant' if is_malignant else 'benign',
            binary_confidence=binary_confidence,
            multiclass_prediction=result['stage2']['condition'] if is_malignant else None,
            multiclass_confidence=result['stage2']['confidence'] if is_malignant else None,
            all_probabilities=str(result['stage2']['all_confidences'] if is_malignant else {}),
            is_malignant_alert=is_malignant
        )
        db.session.add(pred_record)
        db.session.commit()

        return jsonify(result)
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/predict-camera', methods=['POST'])
@login_required
def predict_camera():
    """Two-stage ensemble prediction from camera"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        base64_image = data['image']

        print("Making live camera ensemble prediction...")

        # STAGE 1: Binary classification
        is_malignant, binary_confidence = predict_binary_from_base64(base64_image)
        result = {
            'success': True,
            'stage1': {
                'is_malignant': is_malignant,
                'binary_label': 'Malignant' if is_malignant else 'Benign',
                'binary_confidence': float(binary_confidence),
                'binary_confidence_percent': float(binary_confidence * 100)
            }
        }

        # STAGE 2: Multi-class classification (only if malignant)
        if is_malignant:
            class_name, multiclass_confidence, all_confidences = predict_multiclass_from_base64(base64_image)
            
            # NEW: Add risk assessment
            risk_level, risk_description = get_risk_level(multiclass_confidence)
            
            result['stage2'] = {
                'condition': class_name,
                'confidence': float(multiclass_confidence),
                'confidence_percent': float(multiclass_confidence * 100),
                'risk_level': risk_level,  # NEW
                'risk_description': risk_description,  # NEW
                'all_confidences': all_confidences
            }
            print(f"MALIGNANT ALERT: {class_name} ({multiclass_confidence*100:.2f}%) - {risk_level}")
        else:
            print(f"Benign classification confirmed")

        # Save image and prediction to database
        try:
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            if ',' in base64_image:
                img_data = base64.b64decode(base64_image.split(',')[1])
            else:
                img_data = base64.b64decode(base64_image)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = f"{timestamp}camera_capture.jpg"
            filepath = os.path.join(upload_folder, filename)
            with open(filepath, 'wb') as f:
                f.write(img_data)

            pred_record = Prediction(
                user_id=current_user.id,
                image_path=f'uploads/{filename}',
                binary_prediction='malignant' if is_malignant else 'benign',
                binary_confidence=binary_confidence,
                multiclass_prediction=result['stage2']['condition'] if is_malignant else None,
                multiclass_confidence=result['stage2']['confidence'] if is_malignant else None,
                all_probabilities=str(result['stage2']['all_confidences'] if is_malignant else {}),
                is_malignant_alert=is_malignant
            )
            db.session.add(pred_record)
            db.session.commit()
        except Exception as db_error:
            print(f"Warning: Could not save camera prediction to DB: {db_error}")

        return jsonify(result)
    except Exception as e:
        print(f"❌ Camera prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.created_at.desc()
    ).all()
    return render_template('history.html', predictions=predictions)

@main_bp.route('/delete/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction record"""
    pred = Prediction.query.get_or_404(prediction_id)
    if pred.user_id != current_user.id:
        flash('Unauthorized!', 'danger')
        return redirect(url_for('main.history'))
    try:
        os.remove(f'static/{pred.image_path}')
    except:
        pass
    db.session.delete(pred)
    db.session.commit()
    flash('Prediction deleted successfully!', 'success')
    return redirect(url_for('main.history'))

# ============================================
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
