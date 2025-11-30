from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    """Prediction model for storing inference results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    
    # Prediction results (9-class classification)
    prediction = db.Column(db.String(50), nullable=False)  # Class name (e.g., "melanoma")
    class_idx = db.Column(db.Integer, nullable=True)  # Class index (0-8)
    confidence = db.Column(db.Float, nullable=False)  # Confidence (0-1)
    all_probabilities = db.Column(db.Text, nullable=True)  # JSON string of all class probabilities
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert prediction to dictionary for JSON response"""
        return {
            'id': self.id,
            'prediction': self.prediction,
            'class_idx': self.class_idx,
            'confidence': round(self.confidence * 100, 2),
            'date': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': self.image_path,
            'all_probabilities': self.all_probabilities
        }
