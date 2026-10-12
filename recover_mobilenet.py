
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMAGE_SIZE = 96
NUM_CLASSES = 10

# Rebuild the same architecture used during training.
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights=None
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

weights_path = "models/mobilenet_recovery/model.weights.h5"

print("Loading saved weights...")
model.load_weights(weights_path)
print("SUCCESS: Saved weights loaded!")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

output_path = "models/quickdraw_mobilenet_recovered.keras"
model.save(output_path)

print(f"Recovered model saved to: {output_path}")
