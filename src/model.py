import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes=33):
    """
    Builds the ResNet50 model with a custom classification head.
    """
    print("🏗️  Building ResNet50 Model...")

    # 1. Load Base Model (ResNet50)
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # 2. Freeze Base Layers
    base_model.trainable = False

    # 3. Create Custom Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)  # Flatten 7x7x2048 -> vector of 2048
    x = layers.Dropout(0.5)(x)              # Regularization to prevent overfitting
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # 4. Combine and Compile
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(" Model Architecture Ready.")
    return model

if __name__ == "__main__":
    # Test the model build if run directly
    model = build_model()
    model.summary()
