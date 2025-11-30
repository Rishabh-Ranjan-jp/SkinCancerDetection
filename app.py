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
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create application context and initialize database
with app.app_context():
    db.create_all()

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return '<h1>404 - Page Not Found</h1><a href="/">Go Home</a>', 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return '<h1>500 - Server Error</h1><a href="/">Go Home</a>', 500

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run app
    print("""
    ╔════════════════════════════════════════╗
    ║  SKIN CANCER DETECTION WEB APP        ║
    ║  🚀 Starting Server...                 ║
    ╚════════════════════════════════════════╝
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
