import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIGURATION ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ==================== DATA PREPARATION ====================
def create_sample_dataset():
    """
    Create a sample dataset for demonstration.
    In production, replace with actual HAM10000 or ISIC dataset.
    """
    print("Creating sample dataset...")
    
    os.makedirs('data/train/benign', exist_ok=True)
    os.makedirs('data/train/malignant', exist_ok=True)
    os.makedirs('data/val/benign', exist_ok=True)
    os.makedirs('data/val/malignant', exist_ok=True)
    
    # Generate random images for demo
    for i in range(100):  # 100 benign training samples
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tf.keras.preprocessing.image.save_img(f'data/train/benign/img_{i}.jpg', img)
    
    for i in range(100):  # 100 malignant training samples
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tf.keras.preprocessing.image.save_img(f'data/train/malignant/img_{i}.jpg', img)
    
    for i in range(30):  # 30 validation samples
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tf.keras.preprocessing.image.save_img(f'data/val/benign/img_{i}.jpg', img)
    
    for i in range(30):  # 30 validation samples
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tf.keras.preprocessing.image.save_img(f'data/val/malignant/img_{i}.jpg', img)
    
    print("✓ Sample dataset created")

# ==================== MODEL BUILDING ====================
def build_model():
    """Build transfer learning model using MobileNetV2"""
    print("Building model...")
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base layers
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./127.5, offset=-1),  # Normalize
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    print("✓ Model built successfully")
    return model

# ==================== DATA LOADING ====================
def load_data():
    """Load and preprocess training data"""
    print("Loading data...")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes={'benign': 0, 'malignant': 1}
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes={'benign': 0, 'malignant': 1}
    )
    
    print("✓ Data loaded successfully")
    return train_generator, val_generator

# ==================== TRAINING ====================
def train_model(model, train_gen, val_gen):
    """Train the model"""
    print("Training model...")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1
    )
    
    print("✓ Training complete")
    return history

# ==================== EVALUATION ====================
def evaluate_model(model, val_gen):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Get predictions
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('model/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curve saved as roc_curve.png")

# ==================== MAIN ====================
if __name__ == '__main__':
    # Create sample dataset (replace with real data in production)
    create_sample_dataset()
    
    # Build model
    model = build_model()
    
    # Load data
    train_gen, val_gen = load_data()
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate
    evaluate_model(model, val_gen)
    
    # Save model
    model.save('model/skin_cancer_model.h5')
    print("\n✓ Model saved as model/skin_cancer_model.h5")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
