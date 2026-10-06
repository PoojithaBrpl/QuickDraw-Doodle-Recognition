from data_loader import load_dataset, split_dataset


X, y = load_dataset(max_samples_per_class=1000)

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)


print("\nDataset test successful!")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)