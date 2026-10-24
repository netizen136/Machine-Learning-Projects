# Telco Customer Churn: EDA, Model Training, and FastAPI Serving

This folder takes a customer churn classification problem from exploratory data analysis through model selection and local HTTP inference. It is a self-contained project: the Excel workbook is the source dataset, the fitted preprocessing and classifier are saved as joblib artifacts, and `predict.py` exposes the selected artifacts through FastAPI.

## Project contents

| File or folder | Purpose |
|---|---|
| `Telco_customer_churn.xlsx` | Source customer-level churn dataset used by the notebook and training script. Keep this file beside the scripts. |
| `data_eda.ipynb` | Exploratory data analysis: inspect data quality and categories, visualize numeric distributions, compare features with churn, and explore correlations. |
| `feature_preprocessing.py` | Shared, importable row-level cleanup: drop leakage/identifier fields, map categories, and convert total charges to numeric. |
| `models_fit_predict_bestmodel_save.py` | Build preprocessing/model pipelines, compare candidate classifiers, choose by held-out F1, refit on all rows, and save the winning artifacts. |
| `saved_models/` | One subfolder per training run, containing `preprocessor.joblib` and `classifier.joblib`. These are generated binary artifacts. |
| `predict.py` | Load a saved run, validate prediction requests, and serve `/health` and `/predict` with FastAPI. |
| `requirements.txt` | Python package versions for the serving/training environment. |
| `.gitignore` | Excludes Python caches and generated model artifacts from version control. |

## Dataset and prediction target

The workbook contains the Telco customer churn table. Each row describes one customer, their account/service choices and charges, along with churn outcomes. The training target is **`Churn Value`**: `1` means churned and `0` means retained. The source workbook also includes descriptive and post-outcome columns; those are not all valid model inputs.

The features used for prediction cover:

- **Customer profile:** `Gender`, `Senior Citizen`, `Partner`, and `Dependents`.
- **Account tenure and billing:** `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, and `Total Charges`.
- **Services:** phone service and multiple lines; internet service; online security, backup, device protection, and tech support; streaming TV and movies.

The preprocessing helper removes identifiers/location fields (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) and outcome-derived fields (`Churn Label`, `Churn Score`, `CLTV`, `Churn Reason`). `Churn Value` is separated as the label before preprocessing. Excluding these columns prevents identifiers, redundant target representations, and information learned after the churn outcome from being used as predictors.

The notebook is the place to inspect the actual workbook and understand its distributions before fitting models. It checks sample rows, missing values, distinct category counts, and category distributions; examines numeric histograms; compares numeric features between churn classes with boxplots; calculates category-level customer counts and churn rates; and visualizes numeric correlations, including correlations with churn and highly correlated feature pairs. These analyses describe patterns in this dataset; association in EDA does not establish that a feature causes churn.

## End-to-end flow

```text
Telco_customer_churn.xlsx
          │
          ├── data_eda.ipynb  (inspect, summarize, visualize)
          │
          └── models_fit_predict_bestmodel_save.py
                  ├── split raw X / y
                  ├── fit shared preprocessing + candidate classifier
                  ├── compare held-out metrics and select highest F1
                  └── refit winner on all rows
                           └── saved_models/<run>/
                                 ├── preprocessor.joblib
                                 └── classifier.joblib
                                           │
                                           └── predict.py (load + FastAPI inference)
```

### 1. Explore the data

Open `data_eda.ipynb` in Jupyter and run its cells from top to bottom. The notebook loads the Excel workbook, inspects its shape and contents, checks nulls and cardinality, and explores categorical and numeric features against the churn label. It also introduces a preprocessing pipeline and the model fitting stage. The production training script is `models_fit_predict_bestmodel_save.py`; it keeps the fitted transformations with the model so inference follows the training transformations.

### 2. Apply consistent feature cleanup

`feature_preprocessing.py` defines `DROP_COLUMNS`, `CATEGORY_MAPS`, and `drop_and_map_features(data)`. The function works on a copy, drops columns if present, maps the known text categories to integer codes, and converts `Total Charges` to numeric values (unparseable values become missing and are handled later). Unrecognized category values map to `-1`.

This function is used inside a scikit-learn `FunctionTransformer`, rather than being reimplemented in the API. Since joblib serializes a reference to the function, keep `feature_preprocessing.py` importable alongside `predict.py` when loading a saved preprocessor.

### 3. Fit, compare, and save models

Run the training script from this folder (or invoke it by path from another working directory):

```powershell
python models_fit_predict_bestmodel_save.py
```

The script loads the workbook relative to its own location, so the current working directory does not need to be this folder. It separates `Churn Value` from the raw feature frame and creates a reproducible 80/20 train/test split using `random_state=42` and stratification. Every candidate receives the same training-only-fitted preprocessing pipeline:

- Numerical columns: median imputation, then standard scaling.
- Multi-value categorical columns: most-frequent imputation, then one-hot encoding.
- Binary categories: the shared mapping encodes them as 0/1 before the numeric transformation.

It compares a prior-only `DummyClassifier` baseline with logistic regression, k-nearest neighbors, support vector machine, decision tree, and random forest. The script prints accuracy and F1 for each candidate and a classification report for the selected model. **F1 on the held-out split is the selection criterion**; accuracy is contextual, and the printed held-out metrics are not a separate final test set. After comparison, the selected complete pipeline is refit on all available rows for deployment.

Each execution creates a timestamped directory under `saved_models/`, named for the selected model, and writes two files:

- `preprocessor.joblib`: the fitted cleanup, imputation, scaling, and encoding steps.
- `classifier.joblib`: the fitted classifier selected by held-out F1.

Keep both files from the same run together. Do not load a preprocessor from one run with another run's classifier.

### 4. Serve predictions with FastAPI

The API loads one saved run when `predict.py` starts. By default, `load_model()` sorts run folder names alphabetically and selects the last. This is the script's documented behavior; because the folder names use `DDMMYY_HHMMSS`, alphabetical order is not a reliable chronological ordering across different dates/months. To avoid ambiguity, pass a specific run folder explicitly:

```powershell
python predict.py Support_Vector_Machine_251026_014424
```

To use the default folder:

```powershell
python predict.py
```

Start the API from this folder with Uvicorn instead, if preferred:

```powershell
uvicorn predict:app --host 127.0.0.1 --port 8000
```

`predict.py` binds to `127.0.0.1:8000` when run directly. It provides:

- `GET /health` — returns `{"status": "ok"}` when the app is responding.
- `POST /predict` — accepts one customer object, applies request validation, runs preprocessing and prediction, then returns a binary prediction, human-readable label, and churn probability.
- `/docs` — FastAPI's interactive API documentation while the server is running.

Example request body (JSON property names match the source dataset's column names):

```json
{
  "Gender": "Female",
  "Senior Citizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure Months": 12,
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "DSL",
  "Online Security": "Yes",
  "Online Backup": "No",
  "Device Protection": "No internet service",
  "Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 70.5,
  "Total Charges": 846.0
}
```

The response has the form:

```json
{
  "prediction": 1,
  "label": "Churn",
  "churn_probability": 0.72
}
```

The probability shown is illustrative; actual output depends on the selected saved model. The Pydantic request model limits categorical values to known dataset categories and constrains tenure to 1–75, monthly charges to 60–120, and total charges to 60–9000. Supply all listed fields. The returned probability is the model's estimate for class `1` and should not be interpreted as a guarantee.

## Environment setup

Use Python and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```
 Run training before starting the API if `saved_models/` has no valid run directory. Then do prediction with saved model with user inference data.

```powershell
python models_fit_predict_bestmodel_save.py
python predict.py
```
