# Binary Logistic Regression - Binary Cross-Entropy

Code: Python, NumPy, Pandas, and Matplotlib; logistic regression is implemented
without scikit-learn.

Dataset: Breast Cancer Wisconsin diagnostic data, included as
`breast_cancer_data.csv` (569 cases and 30 numeric input features).

## About

The notebook predicts the `diagnosis` of a breast mass: malignant (1) or
benign (0). The 30 input features describe cell nuclei, including measurements
of radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry, and fractal dimension. Each measurement is provided
as a mean, standard error, or worst value.

## What I am learning

- Preparing labels and exploring the distribution of the two diagnosis classes.
- Splitting data into training and test sets, then standardizing features
  using statistics learned from the training set.
- Implementing the sigmoid function, binary cross-entropy loss, and gradient
  descent with NumPy.
- Converting predicted probabilities into binary class predictions and
  evaluating accuracy on held-out data.
- Connecting model weights and predictions to a practical binary
  classification problem.

## Reference

https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
