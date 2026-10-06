import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# 10 QuickDraw classes for our project
CLASSES = [
    "apple",
    "banana",
    "car",
    "cat",
    "dog",
    "fish",
    "house",
    "star",
    "tree",
    "airplane"
]

DATA_DIR = Path("data")


def load_class_data(class_name, max_samples=10000):
    """
    Load one QuickDraw .npy file and return the selected samples.
    """

    file_path = DATA_DIR / f"{class_name}.npy"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    data = np.load(file_path)

    # Use only the required number of samples
    data = data[:max_samples]

    return data


def load_dataset(max_samples_per_class=10000):
    """
    Load all selected classes and create X and y.
    """

    X_list = []
    y_list = []

    for label, class_name in enumerate(CLASSES):

        print(f"Loading {class_name}...")

        data = load_class_data(
            class_name,
            max_samples=max_samples_per_class
        )

        X_list.append(data)
        y_list.extend(
            [label] * len(data)
        )

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)

    # Normalize pixel values from 0-255 to 0-1
    X = X.astype("float32") / 255.0

    return X, y


def split_dataset(X, y, random_state=42):
    """
    Split data into:
    80% training
    10% validation
    10% testing
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    X, y = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    print("\nDataset loaded successfully!")
    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Testing data:", X_test.shape)