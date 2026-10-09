import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

# Keep the experiment small enough for a laptop.
DATA_DIR = "data"
MODEL_DIR = "models"
CLASSES = [
    "airplane", "apple", "banana", "car", "cat",
    "dog", "fish", "house", "star", "tree"
]
SAMPLES_PER_CLASS = 1000
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# Load a small, balanced dataset.
images = []
labels = []

for label, class_name in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, class_name + ".npy")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Please check that your data files exist."
        )

    drawings = np.load(path, mmap_mode="r")
    count = min(SAMPLES_PER_CLASS, len(drawings))

    images.append(np.asarray(drawings[:count], dtype=np.uint8))
    labels.extend([label] * count)
    print(f"Loaded {count} drawings for {class_name}")

X = np.concatenate(images, axis=0).reshape(-1, 28, 28, 1)
y = np.asarray(labels, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Resize drawings and convert grayscale images to three channels.
def prepare_image(image, label):
    image = tf.image.resize(
        tf.cast(image, tf.float32), (IMAGE_SIZE, IMAGE_SIZE)
    )
    image = tf.image.grayscale_to_rgb(image)
    image = preprocess_input(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE)
train_ds = train_ds.map(prepare_image, num_parallel_calls=2)
train_ds = train_ds.prefetch(1)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.map(prepare_image, num_parallel_calls=2)
test_ds = test_ds.prefetch(1)

# Load the pretrained MobileNetV2 feature extractor.
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining MobileNetV2...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"\nMobileNetV2 test accuracy: {accuracy * 100:.2f}%")

predictions = np.concatenate([
    np.argmax(model.predict(batch_x, verbose=0), axis=1)
    for batch_x, _ in test_ds
])
print("\nClassification report:")
print(classification_report(
    y_test, predictions, target_names=CLASSES, zero_division=0
))

model.save(os.path.join(MODEL_DIR, "quickdraw_mobilenet.keras"))

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training Results")
plt.legend()
plt.tight_layout()
plt.savefig("mobilenet_training_accuracy.png")
plt.close()

print("\nSaved model: models/quickdraw_mobilenet.keras")
print("Saved graph: mobilenet_training_accuracy.png")
