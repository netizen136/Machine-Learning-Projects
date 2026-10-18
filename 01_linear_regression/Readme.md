# Linear Regression - Mean Squared Error

Code: Python, NumPy, Pandas, and Matplotlib; linear regression is implemented
without scikit-learn.

Dataset: Auto MPG dataset, included as `auto-mpg.csv` (398 rows before cleaning).

## About

The notebook predicts a car's fuel economy (`mpg`, miles per gallon) from
vehicle attributes. The input features are cylinders, displacement,
horsepower, weight, acceleration, model year, and origin. The car name is not
used as a model feature, and rows with missing horsepower are removed.

## What I am learning

- Exploring and cleaning a real-world dataset, including handling invalid
  horsepower values.
- Splitting data into training and test sets and standardizing features and
  target values using training-set statistics.
- Implementing linear predictions, mean squared error cost, and gradient
  descent with NumPy.
- Tracking cost during training and using convergence to stop optimization.
- Using the trained parameters to predict fuel economy on held-out data.

## Reference

https://www.kaggle.com/datasets/yasserh/auto-mpg-dataset/data
