# Drawing: A New Way to Search

## QuickDraw Doodle Recognition

A machine learning project for recognizing hand-drawn doodles using the Google QuickDraw dataset.

## Project Overview

This project compares classical machine learning and deep learning approaches for recognizing hand-drawn doodles from the QuickDraw dataset.

The project uses 10 doodle categories:

- Apple
- Banana
- Car
- Cat
- Dog
- Fish
- House
- Star
- Tree
- Airplane

## Dataset

The project uses the Google QuickDraw dataset.

Each class is stored as a `.npy` file containing 28 × 28 grayscale doodle images.

The dataset is split into:

- 80% Training
- 10% Validation
- 10% Testing

Pixel values are normalized from 0–255 to 0–1.

The dataset files are not included in this repository because they are large.

Place the downloaded `.npy` files inside the `data/` folder.

## Classical Machine Learning

The classical ML part compares:

1. Logistic Regression
2. Linear SVM
3. RBF SVM

All models are evaluated using test accuracy and training time.

### Results

| Model | Accuracy | Training Time |
|---|---:|---:|
| Logistic Regression | 69.30% | 12.00 s |
| Linear SVM | 67.00% | 19.67 s |
| RBF SVM | 82.40% | 39.43 s |

RBF SVM achieved the highest accuracy among the classical machine learning models.

## Project Structure

```text
QuickDraw-Doodle-Recognition/
│
├── data/
│   └── QuickDraw .npy files
│
├── src/
│   ├── data_loader.py
│   ├── test_data_loader.py
│   ├── logistic_regression.py
│   ├── svm.py
│   └── evaluation.py
│
├── models/
├── notebooks/
├── app/
├── report/
│
├── .gitignore
└── README.md