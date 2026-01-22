# Training code goes here
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Import the model architecture we defined in model.py
from model import build_model

# --- Configuration ---
DATASET_PATH = 'data'
KAGGLE_DATASET = 'vipoooool/new-plant-diseases-dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def download_dataset():
    """
    Downloads and extracts the dataset from Kaggle.
    Requires kaggle.json to be in the root directory.
    """
    if os.path.exists(os.path.join(DATASET_PATH, 'New Plant Diseases Dataset(Augmented)')):
        print("✅ Dataset already exists. Skipping download.")
        return

    print("⬇️  Downloading dataset from Kaggle... (This may take a while)")
    
    # Ensure data directory exists
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Run Kaggle CLI command
    exit_code = os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_PATH}")
    
    if exit_code != 0:
        raise Exception("❌ Kaggle download failed. Did you place kaggle.json in the root folder?")

    print("📦 Unzipping dataset...")
    zip_path = os.path.join(DATASET_PATH, 'new-plant-diseases-dataset.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_PATH)
    
    print("✅ Dataset ready.")

def train_model():
    """
    Main training pipeline.
    """
    # 1. Prepare Data
    download_dataset()
    
    # Path to the specific folder inside the unzipped data
    base_dir = os.path.join(DATASET_PATH, 'New Plant Diseases Dataset(Augmented)')
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # 2. Data Augmentation (Pre-processing)
    # Rescale pixels to 0-1 and add random variety to training images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    print("🔄 Loading images from directory...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # 3. Build Model
    # We dynamically detect the number of classes (folders) found
    num_classes = train_generator.num_classes
    print(f"🧠 Detected {num_classes} classes. Building ResNet50...")
    
    model = build_model(num_classes=num_classes)

    # 4. Callbacks (Best Practices)
    # Save the best model only
    checkpoint = ModelCheckpoint(
        'model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    # Stop early if accuracy stops improving
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    # 5. Train
    print("🚀 Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("🎉 Training Complete. Best model saved as 'model.h5'")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"❌ Error: {e}")
