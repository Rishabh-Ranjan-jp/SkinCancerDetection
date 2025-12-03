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

    """Prediction model for ensemble two-stage classification"""

    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    image_path = db.Column(db.String(255), nullable=False)

    # Stage 1: Binary Classification (benign/malignant)

    binary_prediction = db.Column(db.String(20), nullable=False)  # 'benign' or 'malignant'

    binary_confidence = db.Column(db.Float, nullable=False)  # 0-1

    # Stage 2: Multi-class Classification (only if malignant)

    multiclass_prediction = db.Column(db.String(50), nullable=True)  # Class name (e.g., "melanoma")

    multiclass_confidence = db.Column(db.Float, nullable=True)  # 0-1

    # Alert flag for malignant cases

    is_malignant_alert = db.Column(db.Boolean, default=False)

    # All probabilities from multi-class model

    all_probabilities = db.Column(db.Text, nullable=True)  # JSON string

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):

        """Convert prediction to dictionary for JSON response"""

        return {

            'id': self.id,

            'binary_prediction': self.binary_prediction,

            'binary_confidence': round(self.binary_confidence * 100, 2),

            'multiclass_prediction': self.multiclass_prediction,

            'multiclass_confidence': round(self.multiclass_confidence * 100, 2) if self.multiclass_confidence else None,

            'is_malignant_alert': self.is_malignant_alert,

            'date': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),

            'image_path': self.image_path,

            'all_probabilities': self.all_probabilities

        }
