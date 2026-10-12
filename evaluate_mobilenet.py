
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASSES = [
    "airplane", "apple", "banana", "car", "cat",
    "dog", "fish", "house", "star", "tree"
]

DATA_DIR = "data"
MODEL_PATH = "models/quickdraw_mobilenet_recovered.keras"
SAMPLES_PER_CLASS = 1000
IMAGE_SIZE = 96
BATCH_SIZE = 32

print("Loading dataset...")

images = []
labels = []

for label, class_name in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, class_name + ".npy")
    drawings = np.load(path, mmap_mode="r")
    count = min(SAMPLES_PER_CLASS, len(drawings))

    images.append(np.asarray(drawings[:count], dtype=np.uint8))
    labels.extend([label] * count)
    print(f"Loaded {class_name}: {count} drawings")

X = np.concatenate(images, axis=0).reshape(-1, 28, 28, 1)
y = np.asarray(labels, dtype=np.int32)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nPreparing MobileNetV2 test images...")

X_test = tf.image.resize(
    tf.cast(X_test, tf.float32),
    (IMAGE_SIZE, IMAGE_SIZE)
)
X_test = tf.image.grayscale_to_rgb(X_test)
X_test = preprocess_input(X_test)

print("\nLoading saved MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating MobileNetV2...")
loss, accuracy = model.evaluate(
    X_test, y_test, batch_size=BATCH_SIZE, verbose=1
)

print(f"\nMobileNetV2 Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test samples: {len(y_test)}")
