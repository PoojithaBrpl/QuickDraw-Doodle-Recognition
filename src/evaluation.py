import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from data_loader import load_dataset, split_dataset, CLASSES


# Load the same dataset used for our classical ML experiments
X, y = load_dataset(max_samples_per_class=1000)

# Split into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ),
    "Linear SVM": SVC(
        kernel="linear",
        random_state=42
    ),
    "RBF SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}


results = []


for name, model in models.items():

    print(f"\nTraining {name}...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Training Time (s)": training_time
        }
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(filename)

    plt.show()


# Print comparison table
print("\n\nModel Comparison")
print("================")

for result in results:
    print(
        f"{result['Model']:<22} "
        f"Accuracy: {result['Accuracy']:.4f}    "
        f"Training Time: {result['Training Time (s)']:.2f}s"
    )

# Show the most confused class pairs for the best model (RBF SVM)
print("\nRBF SVM - Most Confused Class Pairs")
print("===================================")

cm = confusion_matrix(y_test, models["RBF SVM"].predict(X_test))

confused_pairs = []

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append(
                (cm[i, j], CLASSES[i], CLASSES[j])
            )

confused_pairs.sort(reverse=True)

for count, actual, predicted in confused_pairs[:10]:
    print(
        f"Actual: {actual:<10} "
        f"Predicted as: {predicted:<10} "
        f"Count: {count}"
    )