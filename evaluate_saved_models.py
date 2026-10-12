
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

CLASSES = [
    "apple", "banana", "car", "cat", "dog",
    "fish", "house", "star", "tree", "airplane"
]

DATA_DIR = Path("data")
MODEL_PATH = "models/quickdraw_cnn.keras"
SAMPLES_PER_CLASS = 10000

print("Loading test dataset...")

images = []
labels = []

for label, class_name in enumerate(CLASSES):
    path = DATA_DIR / f"{class_name}.npy"
    data = np.load(path, mmap_mode="r")
    data = np.array(data[:SAMPLES_PER_CLASS])

    images.append(data)
    labels.extend([label] * len(data))
    print(f"Loaded {class_name}: {len(data)} drawings")

X = np.concatenate(images, axis=0)
y = np.array(labels)

X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

print("\nLoading saved CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating CNN on test data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\nCNN Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test samples: {len(y_test)}")
