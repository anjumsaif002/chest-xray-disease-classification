import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_augmentation_layer():
    """
    Data augmentation pipeline to prevent overfitting on small medical datasets.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="data_augmentation")

def build_model(num_classes=4, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Builds a Transfer Learning CNN model using MobileNetV2 pre-trained on ImageNet.
    
    Args:
        num_classes (int): Number of target disease categories.
        input_shape (tuple): Shape of input images (H, W, C).
        learning_rate (float): Optimizer learning rate.
        
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = create_augmentation_layer()(inputs)
    
    # Rescale [0, 1] images to [-1, 1] for MobileNetV2
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="chest_disease_mobilenetv2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
