import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_dataset, split_dataset


# Load a smaller dataset for SVM
# SVM is more computationally expensive than Logistic Regression.
X, y = load_dataset(max_samples_per_class=1000)

# Split into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)


def train_and_evaluate(kernel):
    """
    Train an SVM with the given kernel
    and return accuracy and training time.
    """

    model = SVC(
        kernel=kernel,
        random_state=42
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)

    return test_accuracy, training_time


# Linear SVM
linear_accuracy, linear_time = train_and_evaluate("linear")

print("\nLinear SVM Results")
print("------------------")
print(f"Test Accuracy: {linear_accuracy:.4f}")
print(f"Training Time: {linear_time:.2f} seconds")


# RBF SVM
rbf_accuracy, rbf_time = train_and_evaluate("rbf")

print("\nRBF SVM Results")
print("----------------")
print(f"Test Accuracy: {rbf_accuracy:.4f}")
print(f"Training Time: {rbf_time:.2f} seconds")
