"""
House Price Prediction - Model Training Script
Based on the Mini Project Report by Varun Mishra
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prepare_data():
    """Load California Housing dataset and perform basic feature engineering."""
    df = pd.read_csv(DATA_PATH)

    # Feature engineering (as suggested in the report)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Target
    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])

    return X, y


def build_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline: imputation + scaling + one-hot encoding."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(model, X_test, y_test, name="Model"):
    """Compute and print MAE, RMSE, R²."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def plot_results(y_test, y_pred_lr, y_pred_rf, feature_importance, feature_names):
    """Generate evaluation plots and save them."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Actual vs Predicted - Linear Regression
    axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.3, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("Actual Price")
    axes[0, 0].set_ylabel("Predicted Price")
    axes[0, 0].set_title("Linear Regression: Actual vs Predicted")

    # Actual vs Predicted - Random Forest
    axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.3, s=10, color="green")
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0, 1].set_xlabel("Actual Price")
    axes[0, 1].set_ylabel("Predicted Price")
    axes[0, 1].set_title("Random Forest: Actual vs Predicted")

    # Residuals - Random Forest
    residuals = y_test - y_pred_rf
    axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_xlabel("Residual (Actual - Predicted)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Random Forest Residuals")

    # Feature Importance (Random Forest)
    top_n = 10
    indices = np.argsort(feature_importance)[-top_n:]
    axes[1, 1].barh(range(top_n), feature_importance[indices], color="teal")
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 1].set_xlabel("Importance")
    axes[1, 1].set_title("Top Feature Importances (Random Forest)")

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {plot_path}")
    plt.close()


def main():
    print("=" * 60)
    print("House Price Prediction - Training Pipeline")
    print("=" * 60)

    # 1. Load data
    X, y = load_and_prepare_data()
    print(f"\nDataset shape: {X.shape}")

    # Define feature types
    categorical_features = ["ocean_proximity"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 4. Linear Regression Pipeline
    print("\nTraining Linear Regression...")
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, "Linear Regression")

    # 5. Random Forest Pipeline
    print("\nTraining Random Forest Regressor...")
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

    # 6. Feature importance
    # Get feature names after preprocessing
    ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    feature_importance = rf_pipeline.named_steps["regressor"].feature_importances_

    # 7. Save models and artifacts
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "linear_regression.pkl"))
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(
        {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_feature_names": all_feature_names,
            "lr_metrics": {k: v for k, v in lr_metrics.items() if k != "y_pred"},
            "rf_metrics": {k: v for k, v in rf_metrics.items() if k != "y_pred"},
        },
        os.path.join(MODELS_DIR, "metadata.pkl"),
    )
    print(f"\nModels saved to {MODELS_DIR}/")

    # 8. Plots
    plot_results(
        y_test,
        lr_metrics["y_pred"],
        rf_metrics["y_pred"],
        feature_importance,
        all_feature_names,
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
