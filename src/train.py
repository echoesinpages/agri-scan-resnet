import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

# --- Configuration ---
DATASET_PATH = 'data'
KAGGLE_DATASET = 'vipoooool/new-plant-diseases-dataset'
IMG_SIZE = (224, 224) # Increased from 150x150 to match ResNet standard
BATCH_SIZE = 32
EPOCHS = 10

def download_dataset():
    """
    Auto-downloads dataset using your kaggle.json
    """
    # Check if folder exists to avoid re-downloading
    if os.path.exists(os.path.join(DATASET_PATH, 'New Plant Diseases Dataset(Augmented)')):
        print("✅ Dataset found. Skipping download.")
        return

    print("⬇️  Downloading dataset... (Ensure kaggle.json is in root)")
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Use Kaggle API
    exit_code = os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_PATH}")
    if exit_code != 0:
        raise Exception("❌ Download failed. Check kaggle.json!")

    # Unzip
    print("📦 Unzipping...")
    with zipfile.ZipFile(os.path.join(DATASET_PATH, 'new-plant-diseases-dataset.zip'), 'r') as z:
        z.extractall(DATASET_PATH)

def train_model():
    download_dataset()
    
    # Define paths
    base_dir = os.path.join(DATASET_PATH, 'New Plant Diseases Dataset(Augmented)')
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # Data Generators (Replaces your manual 'for' loops)
    # 1. Training: With Augmentation (Flip, Rotate, Zoom)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 2. Validation: Just Rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)

    print("🔄 Loading Images...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Build & Train
    model = build_model(num_classes=train_generator.num_classes)

    checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3)

    print("🚀 Starting Training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=[checkpoint, early_stop]
    )
    print("🎉 Done! Best model saved to model.h5")

if __name__ == "__main__":
    train_model()
