
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Drawing classes
CLASSES = [
    "apple", "banana", "car", "cat", "dog",
    "fish", "house", "star", "tree", "airplane"
]

DATA_DIR = Path("data")
SAMPLES_PER_CLASS = 10000
EPOCHS = 10
BATCH_SIZE = 128

# Load a small number of drawings from each class
def load_dataset():
    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        file_path = DATA_DIR / f"{class_name}.npy"
        print(f"Loading {class_name}...")

        data = np.load(file_path, mmap_mode="r")
        data = np.array(data[:SAMPLES_PER_CLASS])

        images.append(data)
        labels.extend([label] * len(data))

    X = np.concatenate(images, axis=0)
    y = np.array(labels)

    # Convert 784 pixels into 28 x 28 images
    X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return X, y

# Load and split the data
X, y = load_dataset()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

# Create the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the CNN
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
Path("models").mkdir(exist_ok=True)
model.save("models/quickdraw_cnn.keras")
print("Model saved to models/quickdraw_cnn.keras")

# Save a training accuracy graph
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.savefig("cnn_training_accuracy.png")
print("Graph saved to cnn_training_accuracy.png")

# Detailed classification results
predictions = model.predict(X_test, verbose=1).argmax(axis=1)
print(classification_report(y_test, predictions, target_names=CLASSES))
