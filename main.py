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

main_bp = Blueprint('main', __name__)

# ============================================
# CONFIGURATION
# ============================================

# Class mapping from notebook training
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
MODEL_PATH = 'models/best_model_fold3.pt'  # Update fold number if using different fold
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation (exactly as in notebook validation)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Global model variable
MODEL = None

# ============================================
# MODEL LOADING
# ============================================

def load_model():
    """Load EfficientNet-B0 model with trained weights"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        
        # Create base EfficientNet-B0 model
        model = efficientnet_b0(weights=None)  # Don't load default weights
        
        # Replace classifier for 9 classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        MODEL = model
        print("✓ Model loaded successfully!")
        return MODEL
        
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}")
        print("Please ensure your best model checkpoint is saved at:", MODEL_PATH)
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def preprocess_image(image_path):
    """
    Preprocess image for model inference
    Matches notebook validation preprocessing
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = INFERENCE_TRANSFORM(img)  # Returns tensor [3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)   # Add batch dimension [1, 3, 224, 224]
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def predict_image(image_path):
    """
    Make prediction on image using trained model
    
    Returns:
        class_name (str): Predicted skin condition
        confidence (float): Confidence score (0-1)
        class_idx (int): Class index (0-8)
        all_confidences (dict): All class confidences
    """
    model = load_model()
    
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)  # Raw logits
        probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities [0-1]
        confidence, class_idx = torch.max(probabilities, 1)
        
        # Convert to numpy/python types
        confidence = confidence.item()  # Single float value
        class_idx = class_idx.item()    # Single integer
        class_name = CLASS_NAMES[class_idx]
        
        # Get all class confidences for detailed output
        all_probs = probabilities[0].cpu().numpy()
        all_confidences = {
            CLASS_NAMES[i]: float(all_probs[i])
            for i in range(NUM_CLASSES)
        }
    
    return class_name, confidence, class_idx, all_confidences

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
    Handle prediction request
    
    Expected POST data:
        - image: File object (jpg/png/jpeg/bmp/gif)
    
    Returns JSON:
        {
            'success': bool,
            'prediction': str (class name),
            'confidence': float (0-1),
            'confidence_percent': float (0-100),
            'class_idx': int (0-8),
            'all_confidences': {class_name: probability, ...},
            'image_path': str,
            'error': str (if failed)
        }
    """
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Create upload directory
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save uploaded file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = secure_filename(f"{timestamp}{file.filename}")
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Make prediction
        print(f"Making prediction for: {filename}")
        class_name, confidence, class_idx, all_confidences = predict_image(filepath)
        confidence_percent = confidence * 100
        
        # Save prediction to database
        pred_record = Prediction(
            user_id=current_user.id,
            image_path=f'uploads/{filename}',
            prediction=class_name,
            confidence=confidence,
            class_idx=class_idx,
            all_probabilities=str(all_confidences)  # Store as string for database
        )
        
        db.session.add(pred_record)
        db.session.commit()
        
        print(f"Prediction: {class_name} ({confidence_percent:.2f}%)")
        
        return jsonify({
            'success': True,
            'prediction': class_name,
            'confidence': float(confidence),
            'confidence_percent': float(confidence_percent),
            'class_idx': int(class_idx),
            'all_confidences': all_confidences,
            'image_path': f'uploads/{filename}'
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.created_at.desc()
    ).all()
    return render_template('history.html', predictions=predictions)

@main_bp.route('/delete/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction record"""
    pred = Prediction.query.get_or_404(prediction_id)
    
    if pred.user_id != current_user.id:
        flash('Unauthorized!', 'danger')
        return redirect(url_for('main.history'))
    
    # Delete image file
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
