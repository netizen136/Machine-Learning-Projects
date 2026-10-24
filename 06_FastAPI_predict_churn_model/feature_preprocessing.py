import pandas as pd

# We are keeping feature preprocessing in a separate module so that 
# the same drop, category mapping, and numeric conversion can be applied 
# to both training and prediction data. This ensures that the model sees 
# the same features in the same format during inference as it did during training.

# model_prediction.py saves the preprocessor as a joblib artifact.
# preprocessor.joblib does not save this manually written function but points to this function.
# While prediction, predict.py has to load that preprocessor.joblib to apply the same 
# transformations to new data, which in turn again needs this function 
# to be importable so that joblib can resolve the reference.

DROP_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Churn Label", "Churn Score",
    "CLTV", "Churn Reason",
]

CATEGORY_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Senior Citizen": {"No": 0, "Yes": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "Phone Service": {"No": 0, "Yes": 1},
    "Multiple Lines": {"No": 0, "Yes": 1, "No phone service": 2},
    "Internet Service": {"No": 0, "DSL": 1, "Fiber optic": 2},
    "Online Security": {"No": 0, "Yes": 1, "No internet service": 2},
    "Online Backup": {"No": 0, "Yes": 1, "No internet service": 2},
    "Device Protection": {"No": 0, "Yes": 1, "No internet service": 2},
    "Tech Support": {"No": 0, "Yes": 1, "No internet service": 2},
    "Streaming TV": {"No": 0, "Yes": 1, "No internet service": 2},
    "Streaming Movies": {"No": 0, "Yes": 1, "No internet service": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "Paperless Billing": {"No": 0, "Yes": 1},
    "Payment Method": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    },
}


def drop_and_map_features(data):
    data = data.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    for column, mapping in CATEGORY_MAPS.items():
        data[column] = data[column].map(mapping).fillna(-1).astype(int)

    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    return data
