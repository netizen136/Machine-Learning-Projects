# Multiclass Logistic Regression - Softmax Cross-Entropy

Code: Python, NumPy, Pandas, and Matplotlib; the classifier is implemented
without scikit-learn.

Dataset: Wine Quality dataset, included as `winequalityN.csv` (6,497 rows).

## About

The notebook predicts wine quality ratings from 3 to 9 using 12 input
features: wine type (white or red), fixed and volatile acidity, citric acid,
residual sugar, chlorides, free and total sulfur dioxide, density, pH,
sulphates, and alcohol. Wine type is encoded as a numeric feature.

## What I am learning

- Exploring and preparing a multiclass dataset and standardizing features.
- Mapping quality labels to class indices and one-hot encoding target labels.
- Implementing numerically stable softmax, cross-entropy loss, and gradient
  descent with NumPy.
- Producing class probabilities, converting them to quality predictions, and
  checking accuracy on held-out data.
- Understanding how a linear classifier extends from two classes to multiple
  classes.

## Reference

https://www.kaggle.com/code/mgmarques/wines-type-and-quality-classification-exercises/notebook
