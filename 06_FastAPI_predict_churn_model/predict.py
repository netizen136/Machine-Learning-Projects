"""FastAPI endpoint for the joblib artifacts created by model_prediction.py."""

import argparse
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
# Joblib needs feature_preprocessing module available to resolve the FunctionTransformer stored in preprocessor.joblib.
from feature_preprocessing import drop_and_map_features
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
from typing import Literal
from sklearn.pipeline import Pipeline
import uvicorn


SAVED_MODELS = Path(__file__).resolve().parent / "saved_models"

def load_model(run_name=None):
    """Load one saved run and return its complete inference pipeline and folder.

    Args:
        run_name: Optional folder name inside saved_models. When omitted, use
            the alphabetically last run folder.

    Raises:
        FileNotFoundError: If no run folders exist when choosing the default.
            If an explicit folder or artifact is missing, joblib raises this
            error while opening the requested artifact.
    """
    # If the caller supplied a folder name, use that run directly. Otherwise
    # find the run folders, sort their names alphabetically, and select the last.
    model_dir = SAVED_MODELS / run_name if run_name else None
    if model_dir is None:
        run_dirs = sorted(path for path in SAVED_MODELS.iterdir() if path.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"No saved model run found in {SAVED_MODELS}")
        model_dir = run_dirs[-1]

    # Training saves preprocessing and classification as separate joblib files.
    # Load both pieces from the same run folder so they stay paired correctly.
    loaded_preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    loaded_classifier = joblib.load(model_dir / "classifier.joblib")

    # Rebuild the original inference order: first preprocess the raw feature
    # DataFrame, then pass the transformed features to the classifier.
    model = Pipeline([
        ("preprocessor", loaded_preprocessor),
        ("classifier", loaded_classifier),
    ])
    return model, model_dir


# Load a default run when this module is imported by FastAPI/uvicorn. Direct
# command-line startup can replace it below when the caller names another run.
model, loaded_model_dir = load_model()

# FastAPI uses this object to serve the health and prediction routes.
app = FastAPI(title="Churn Prediction API")

# input_columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
#        'Phone Service', 'Multiple Lines', 'Internet Service',
#        'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
#        'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
#        'Payment Method', 'Monthly Charges', 'Total Charges']

class PredictionRequest(BaseModel):
    # Pydantic validates each submitted field before predict() is called.
    # Field aliases retain the exact column names expected by the trained model.
    Gender: Annotated[Literal["Female", "Male"], Field(description="Gender of the customer")]
    Senior_Citizen: Annotated[Literal["No", "Yes"], Field(alias="Senior Citizen", description="Whether the customer is a senior citizen")]
    Partner: Annotated[Literal["No", "Yes"], Field(description="Whether the customer has a partner")]
    Dependents: Annotated[Literal["No", "Yes"], Field(description="Whether the customer has dependents")]
    Tenure_Months: Annotated[int, Field(alias="Tenure Months", ge=1, le=75, description="Tenure in months in range 1-75")]
    Phone_Service: Annotated[Literal["No", "Yes"], Field(alias="Phone Service", description="Whether the customer has phone service")]
    Multiple_Lines: Annotated[Literal["No", "Yes", "No phone service"], Field(alias="Multiple Lines", description="Whether the customer has multiple lines")]
    Internet_Service: Annotated[Literal["No", "DSL", "Fiber optic"], Field(alias="Internet Service", description="Type of internet service")]
    Online_Security: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Online Security", description="Whether the customer has online security")]
    Online_Backup: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Online Backup", description="Whether the customer has online backup")]
    Device_Protection: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Device Protection", description="Whether the customer has device protection")]
    Tech_Support: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Tech Support", description="Whether the customer has tech support")]
    Streaming_TV: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Streaming TV", description="Whether the customer has streaming TV")]
    Streaming_Movies: Annotated[Literal["No", "Yes", "No internet service"], Field(alias="Streaming Movies", description="Whether the customer has streaming movies")]
    Contract: Annotated[Literal["Month-to-month", "One year", "Two year"], Field(description="Type of contract")]
    Paperless_Billing: Annotated[Literal["No", "Yes"], Field(alias="Paperless Billing", description="Whether the customer has paperless billing")]
    Payment_Method: Annotated[Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], Field(alias="Payment Method", description="Payment method")]
    Monthly_Charges: Annotated[float, Field(alias="Monthly Charges", ge=60, le=120, description="Monthly charges in range 60-120")]
    Total_Charges: Annotated[float, Field(alias="Total Charges", ge=60, le=9000, description="Total charges in range 60-9000")]



@app.get("/health")
def health():
    """Simple liveness endpoint for checking that the API is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Return a churn label and probability for one validated customer row.

    model_dump(by_alias=True) turns Python-safe field names back into the
    dataset's original column names. The saved preprocessing pipeline then
    performs the same feature cleanup and transformations used in training.
    """
    row = pd.DataFrame([request.model_dump(by_alias=True)])
    # Pipeline.predict and predict_proba apply preprocessing automatically.
    prediction = int(model.predict(row)[0])
    churn_probability = float(
        model.predict_proba(row)[0][list(model.classes_).index(1)]
    )

    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No churn",
        "churn_probability": churn_probability,
    }


if __name__ == "__main__":
    # Direct invocation accepts an optional saved_models subfolder name:
    # `python predict.py` uses the alphabetically last run, while
    # `python predict.py <folder>` loads the named run.
    parser = argparse.ArgumentParser(description="Run the churn prediction API.")
    parser.add_argument(
        "model_folder",
        nargs="?",
        help="Saved model folder name under saved_models (defaults to latest).",
    )
    args = parser.parse_args()
    if args.model_folder:
        # Replace the module-level default with the explicitly requested run.
        model, loaded_model_dir = load_model(args.model_folder)
    print(f"Using model artifacts from {loaded_model_dir}")
    # Start the local API server using the selected model pipeline.
    uvicorn.run(app, host="127.0.0.1", port=8000)
