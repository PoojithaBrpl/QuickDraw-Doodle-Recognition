import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_loader import load_dataset, split_dataset


# Load dataset
X, y = load_dataset(max_samples_per_class=1000)

# Split into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)


# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)


# Train the model and measure training time
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time


# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)


# Evaluate on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# Display results
print("\nLogistic Regression Results")
print("---------------------------")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Time: {training_time:.2f} seconds")