import os
import sys
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input

# Allow running from project root: python src/train.py
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model


DATASET_PATH = 'data'
KAGGLE_DATASET = 'vipoooool/new-plant-diseases-dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def download_dataset():
    """
    Auto-downloads dataset using your kaggle.json placed in the project root.
    """
    if os.path.exists(os.path.join(DATASET_PATH, 'New Plant Diseases Dataset(Augmented)')):
        print("Dataset already found. Skipping download.")
        return

    print("Downloading dataset... (Ensure kaggle.json is in root)")
    os.makedirs(DATASET_PATH, exist_ok=True)

    exit_code = os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_PATH}")
    if exit_code != 0:
        raise Exception("Download failed. Check that kaggle.json is in the project root.")

    print("Unzipping...")
    zip_path = os.path.join(DATASET_PATH, 'new-plant-diseases-dataset.zip')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATASET_PATH)
    print("Dataset ready.")


def train_model():
    download_dataset()

    base_dir = os.path.join(
        DATASET_PATH,
        'New Plant Diseases Dataset(Augmented)',
        'New Plant Diseases Dataset(Augmented)'
    )

    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # FIX: Use preprocess_input (ImageNet mean subtraction) to match app.py inference.
    # Do NOT use rescale=1./255 — it conflicts with resnet50.preprocess_input used at inference.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print(f"Loading training images from: {train_dir}")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"Loading validation images from: {valid_dir}")
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"Found {train_generator.num_classes} classes.")

    # Build Model
    model = build_model(num_classes=train_generator.num_classes)

    callbacks = [
        ModelCheckpoint('model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
    ]

    print("Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks
    )

    # Final evaluation on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(valid_generator)
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Validation Loss:     {val_loss:.4f}")
    print("\nDone! Best model saved to model.h5")

    return history


if __name__ == "__main__":
    train_model()
