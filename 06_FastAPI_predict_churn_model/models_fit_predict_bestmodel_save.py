from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from feature_preprocessing import CATEGORY_MAPS, DROP_COLUMNS, drop_and_map_features
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
# Read the training dataset relative to this script, so the script works from
# any current working directory as long as the spreadsheet stays beside it.
df = pd.read_excel(ROOT / "Telco_customer_churn.xlsx")

# This transformer calls the importable function in feature_preprocessing.py.
# Keeping it in the saved pipeline means predict.py can apply the same drop,
# category mapping, and numeric conversion used during training.
drop_and_map_transformer = FunctionTransformer(drop_and_map_features, validate=False)

# Choose the columns that remain after drop_and_map_features removes DROP_COLUMNS.
# Binary categories already map to 0/1, so only fields with three or more
# categories are one-hot encoded.
feature_columns = [column for column in df.columns if column not in ["Churn Value", *DROP_COLUMNS]]
categorical_features = [column for column, mapping in CATEGORY_MAPS.items() if len(mapping) > 2]
numeric_features = [column for column in feature_columns if column not in categorical_features]

# Numeric columns get missing-value imputation followed by scaling. Both steps
# learn their statistics only when the pipeline is fitted on training data.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# The multi-value categorical columns get missing-value imputation followed by
# one-hot encoding. The encoder learns its category columns during fit.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # OneHotEncoder defaults to handle_unknown="error".
    ("encoder", OneHotEncoder()),
])

# Apply separate transformations to the numeric and categorical column groups.
# ColumnTransformer returns their transformed values together as one feature set.
column_transformer = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
# Run row-level cleanup first, then imputation/scaling/encoding. This complete
# preprocessing pipeline is fitted as part of each model pipeline.
preprocessor = Pipeline([
    ("drop_and_map_transformer", drop_and_map_transformer),
    ("column_transformer", column_transformer),
])

# Separate the target label from the input features. The target is never passed
# into preprocessing, so it cannot accidentally become a model input.
X = df.drop(columns=["Churn Value"])
y = df["Churn Value"]

# Keep 20% aside for comparing models. Stratification keeps the churn/no-churn
# proportions similar in both splits; random_state makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Candidate classifiers are evaluated with the same split and preprocessing.
# class_weight helps account for the unequal number of churn and non-churn rows.
classifiers = {
    "Baseline": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(
        probability=True, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=8
    ),
}

# Track the F1 score and corresponding fitted pipeline for the best candidate.
# The fitted pipeline includes both preprocessing and classification steps.
results = []
best_f1 = -1
best_model = None
best_model_name = None

for name, classifier in classifiers.items():
    # Pipeline.fit fits preprocessing using X_train, then fits the classifier
    # on the transformed training rows. predict transforms X_test with those
    # already-fitted preprocessing steps before producing predictions.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # F1 is used to select the best model; accuracy is also printed for context.
    f1score = f1_score(y_test, predictions, zero_division=0)
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, predictions), "F1": f1score})
    if f1score > best_f1:
        best_f1 = f1score
        best_model = model
        best_model_name = name

# Display scores for every candidate, followed by detailed metrics for the
# selected candidate on the held-out split.
print(pd.DataFrame(results).set_index("Model").sort_values("F1", ascending=False))
print("\nBest model report:")
print(classification_report(y_test, best_model.predict(X_test)))

# Evaluation above uses the held-out split. Refit the selected pipeline on all
# available rows so the saved model can learn from the full dataset.
best_model.fit(X, y)

# Each run gets its own folder under saved_models, with the selected model name
# and a timestamp to distinguish its artifacts from previous training runs.
saved_models = ROOT / "saved_models"
saved_models.mkdir(exist_ok=True)
run_name = f"{best_model_name.replace(' ', '_')}_{datetime.now().strftime('%d%m%y_%H%M%S')}"
run_dir = saved_models / run_name
run_dir.mkdir()
# Save the fitted preprocessing pipeline and classifier separately. predict.py
# loads both and joins them into a Pipeline for inference on raw request rows.
joblib.dump(best_model.named_steps["preprocessor"], run_dir / "preprocessor.joblib")
joblib.dump(best_model.named_steps["classifier"], run_dir / "classifier.joblib")

print(f"\nSaved best classifier and preprocessor to {run_dir}")
