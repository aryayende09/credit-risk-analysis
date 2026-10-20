
import getpass
from pathlib import Path

import joblib

import pandas as pd
from sqlalchemy import create_engine, URL

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]

ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "outputs"
MODEL_DIR = ML_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

numerical_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

features = numerical_features + categorical_features
target = "loan_status"


def make_preprocessor():
    """Create a separate preprocessing pipeline for each model."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def select_threshold(model, X_validation, y_validation):
    """Choose the tested threshold with the highest validation F1."""

    probabilities = model.predict_proba(X_validation)[:, 1]
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_validation, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_validation, predictions, zero_division=0
                ),
                "F1": f1_score(
                    y_validation, predictions, zero_division=0
                ),
                "Predicted_Defaults": int(predictions.sum()),
            }
        )

    threshold_table = pd.DataFrame(results)

    # If F1 ties, retain the first threshold in THRESHOLDS.
    best_index = threshold_table["F1"].idxmax()
    best_threshold = float(
        threshold_table.loc[best_index, "Threshold"]
    )

    return best_threshold, threshold_table


def evaluate_model(
    model_name, model, threshold, X_test, y_test
):
    """Evaluate a model on the test data."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Average_Precision": average_precision_score(
            y_test, probabilities
        ),
    }

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("\n" + "=" * 65)
    print(f"{model_name.upper()} — TEST RESULTS")
    print(f"Selected threshold: {threshold:.2f}")
    print("=" * 65)

    print("\nConfusion Matrix:")
    print(matrix)
    print("Rows: actual [No Default, Default]")
    print("Columns: predicted [No Default, Default]")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Default", "Default"],
            zero_division=0,
        )
    )

    print(f"ROC-AUC: {metrics['ROC_AUC']:.4f}")
    print(
        "Average Precision: "
        f"{metrics['Average_Precision']:.4f}"
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=["Actual_No_Default", "Actual_Default"],
        columns=["Predicted_No_Default", "Predicted_Default"],
    )
    matrix_table.to_csv(
        OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.csv"
    )

    return metrics, probabilities, predictions


# ============================================================
# 2. LOAD DATA FROM MYSQL
# ============================================================

print("Loading data from MySQL...")

password = getpass.getpass("Enter your MySQL root password: ")

connection_url = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password=password,
    host="localhost",
    port=3306,
    database="credit_risk",
)

engine = create_engine(connection_url)

try:
    df = pd.read_sql("SELECT * FROM loans_cleaned", engine)

    print(f"Rows loaded: {len(df):,}")

    required_columns = features + [target]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ========================================================
    # 3. SELECT PRE-APPROVAL FEATURES AND TARGET
    # ========================================================

    X = df[features].copy()

    # Convert numeric-looking database values to numbers.
    # Unreadable numeric values become missing and are imputed
    # inside the training pipeline.
    for column in numerical_features:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    y = pd.to_numeric(df[target], errors="raise")

    if y.isnull().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "loan_status must contain only non-missing 0 and 1."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Both target classes (0 and 1) must be present."
        )

    # Normalize missing categorical values for the imputer.
    for column in categorical_features:
        X[column] = X[column].astype("object")
        X[column] = X[column].where(X[column].notna(), float("nan"))

    print(f"Input features: {len(features)}")
    print("Target: loan_status")
    print("0 = No Default | 1 = Default")
    print("Excluded: loan_id, loan_grade, loan_int_rate")

    # ========================================================
    # 4. SPLIT: 60% TRAIN, 20% VALIDATION, 20% TEST
    # ========================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nDataset split:")
    print(f"Training:   {len(X_train):,}")
    print(f"Validation: {len(X_validation):,}")
    print(f"Testing:    {len(X_test):,}")

    # ========================================================
    # 5. DEFINE BOTH MODELS
    # ========================================================

    models = {
        "Logistic_Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000),
                ),
            ]
        ),
        "Random_Forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    # ========================================================
    # 6. TRAIN, SELECT THRESHOLDS, AND EVALUATE
    # ========================================================

    comparison_results = []
    validation_threshold_tables = {}
    trained_models = {}
    selected_thresholds = {}
    saved_model_paths = {}

    for model_name, model in models.items():
        print("\n" + "#" * 65)
        print(f"TRAINING: {model_name}")
        print("#" * 65)

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        # Choose a threshold using validation data only.
        threshold, threshold_table = select_threshold(
            model, X_validation, y_validation
        )

        selected_thresholds[model_name] = threshold
        validation_threshold_tables[model_name] = threshold_table

        # Save the complete fitted pipeline and its selected threshold.
        # The pipeline includes preprocessing and the trained classifier.
        model_bundle = {
            "model_name": model_name,
            "pipeline": model,
            "threshold": threshold,
            "features": features,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "target": target,
            "class_labels": {0: "No Default", 1: "Default"},
            "excluded_columns": ["loan_id", "loan_grade", "loan_int_rate"],
            "prediction_scenario": "Intended pre-approval default-risk estimate",
            "random_state": RANDOM_STATE,
        }
        model_path = MODEL_DIR / f"{model_name.lower()}.joblib"
        joblib.dump(model_bundle, model_path)
        saved_model_paths[model_name] = model_path
        print(f"Saved trained pipeline: {model_path}")

        print("\nValidation threshold comparison:")
        print(threshold_table.round(4).to_string(index=False))
        print(f"Selected threshold: {threshold:.2f}")

        # Test data is used for evaluation, not threshold selection.
        metrics, _, _ = evaluate_model(
            model_name,
            model,
            threshold,
            X_test,
            y_test,
        )

        comparison_results.append(metrics)

    # ========================================================
    # 7. SAVE MODEL COMPARISON AND THRESHOLD RESULTS
    # ========================================================

    comparison_table = pd.DataFrame(comparison_results)
    comparison_table = comparison_table.sort_values(
        "F1", ascending=False
    )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False)

    threshold_tables = []

    for model_name, table in validation_threshold_tables.items():
        saved_table = table.copy()
        saved_table.insert(0, "Model", model_name)
        threshold_tables.append(saved_table)

    all_thresholds = pd.concat(
        threshold_tables, ignore_index=True
    )
    threshold_path = OUTPUT_DIR / "validation_threshold_results.csv"
    all_thresholds.to_csv(threshold_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON — TEST DATA")
    print("=" * 65)
    print(comparison_table.round(4).to_string(index=False))

    # ========================================================
    # 8. RANDOM FOREST PERMUTATION FEATURE IMPORTANCE
    # ========================================================

    print("\nCalculating Random Forest feature importance...")
    print("Using validation data; this may take a little time.")

    forest = trained_models["Random_Forest"]

    importance = permutation_importance(
        forest,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance.importances_mean,
            "Std_Dev": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)

    importance_path = (
        OUTPUT_DIR / "random_forest_feature_importance.csv"
    )
    importance_table.to_csv(importance_path, index=False)

    print("\nRandom Forest Feature Importance:")
    print(importance_table.round(5).to_string(index=False))

    # ========================================================
    # 9. COMPLETION SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("PROJECT RUN COMPLETED")
    print("=" * 65)
    print(f"Model comparison: {comparison_path}")
    print(f"Threshold results: {threshold_path}")
    print(f"Feature importance: {importance_path}")
    print(f"CSV outputs directory: {OUTPUT_DIR}")
    print(f"Saved model directory: {MODEL_DIR}")
    for model_name, model_path in saved_model_paths.items():
        print(f"{model_name} model: {model_path}")

finally:
    engine.dispose()
    print("\nMySQL connection closed.")
