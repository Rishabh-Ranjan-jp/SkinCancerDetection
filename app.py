"""
Skin Cancer Detection Web Application
Main Flask Application Entry Point
"""

import os
from flask import Flask
from flask_login import LoginManager
from config import Config
from models import db, User
from auth import auth_bp
from main import main_bp

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(main_bp)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create application context and initialize database
with app.app_context():
    db.create_all()
    print("✓ Database initialized")

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return 'Page not found', 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return 'Internal server error', 500

if __name__ == '__main__':
    # Create upload folder
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("🏥 SKIN CANCER DETECTION WEB APPLICATION")
    print("=" * 60)
    print("✓ Starting Flask application...")
    print("📍 Server: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
