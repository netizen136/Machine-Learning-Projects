# K-Means Clustering

Code: Python, NumPy, Pandas, and Matplotlib; K-means is implemented without
scikit-learn.

Dataset: Mall Customers dataset, included as `Mall_Customers.csv` (200
customers).

## About

This is an unsupervised learning project: K-means groups customers by
similarity rather than predicting a labelled outcome. The notebook clusters
customers using gender, age, annual income (in thousands of dollars), and
spending score (1-100). The unique `CustomerID` column is excluded.

## What I am learning

- Exploring the customer data and encoding gender as a numeric feature.
- Implementing K-means assignment and centroid-update steps with NumPy,
  including convergence checks and empty-cluster handling.
- Measuring cluster fit with inertia and seeing how random centroid
  initialization can lead to different results.
- Repeating clustering with multiple initializations and selecting the run
  with the lowest inertia.
- Using the elbow method and a manually calculated silhouette score to assess
  cluster choices.
- Assigning a new customer to its nearest learned centroid.

## Reference

https://www.kaggle.com/code/abdallahwagih/customers-segmentation-unsupervised-techniques
