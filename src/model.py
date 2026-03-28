import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_model(num_classes=38):
    """
    Builds the ResNet50 model with a custom classification head.
    Input Shape: (224, 224, 3)
    Preprocessing: Uses ResNet50's built-in preprocess_input (ImageNet mean subtraction).
    """
    print("Building ResNet50 Model...")

    # 1. Load Base Model (ResNet50)
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # 2. Freeze Base Layers
    base_model.trainable = False

    # 3. Create Custom Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # 4. Compile
    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
