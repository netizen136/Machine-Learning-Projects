# Titanic Survival Classification

Code: Python, scikit-learn, Pandas, and NumPy.

Dataset: Titanic passenger data from the Kaggle challenge, included as
`titanic_train.csv` and `titanic_test.csv`.

## About

The notebook predicts `Survived`: whether a passenger lived (1) or did not
survive (0). It uses passenger class (`Pclass`), sex, age, number of siblings
or spouses aboard (`SibSp`), number of parents or children aboard (`Parch`),
fare, cabin, and port of embarkation (`Embarked`). Passenger ID, name, and
ticket are excluded from the model features.

## What I am learning

- Exploring passenger features, encoding categorical values, and handling
  missing data.
- Understanding how preprocessing the full dataset before splitting can cause
  data leakage and overly optimistic evaluation.
- Avoiding preprocessing leakage by calculating imputation and scaling values
  from the training split, then applying them to the test split.
- Comparing Logistic Regression, KNN, Decision Tree, Random Forest, Gradient
  Boosting, SVM, and Naive Bayes classifiers.
- Evaluating predictions using accuracy, precision, recall, F1, and ROC-AUC.
- Considering why the choice of evaluation metric matters when comparing
  classifiers.

## Reference

https://www.kaggle.com/c/titanic
