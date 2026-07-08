# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 15 train a RandomForest model for ML-based quant signals
# Target: Predict whether forward 5-day return is greater than 1%

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "models/datasets/ml_signal_dataset.csv"
OUTPUT_FOLDER = "models/results"
CHART_FOLDER = "reports/charts"

TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ma20_ma60_ratio",
    "close_ma20_ratio",
    "close_ma60_ratio",
    "volatility_5d",
    "volatility_20d",
    "rsi"
]

TARGET_COLUMN = "target"


def load_dataset(file_path):
    """
    Load the ML signal dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    return df


def split_train_test(df):
    """
    Split the dataset by time to avoid look-ahead bias.
    Train: up to 2023-12-31
    Test: from 2024-01-01
    """

    train_df = df[df["Date"] <= TRAIN_END_DATE].copy()
    test_df = df[df["Date"] >= TEST_START_DATE].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return train_df, test_df, x_train, y_train, x_test, y_test


def train_random_forest(x_train, y_train):
    """
    Train a RandomForest classification model.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(x_train, y_train)

    return model


def save_feature_importance(model):
    """
    Save feature importance results and chart.
    """

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values("Importance", ascending=False)

    importance_output_path = f"{OUTPUT_FOLDER}/random_forest_feature_importance.csv"
    importance_df.to_csv(importance_output_path, index=False)

    ax = importance_df.plot(
        kind="bar",
        x="Feature",
        y="Importance",
        legend=False,
        figsize=(12, 6)
    )

    plt.title("RandomForest Feature Importance", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Feature", fontsize=11)
    plt.ylabel("Importance", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    chart_output_path = f"{CHART_FOLDER}/random_forest_feature_importance.png"
    plt.savefig(chart_output_path, dpi=300)
    plt.close()

    print(f"Saved feature importance to: {importance_output_path}")
    print(f"Saved chart to: {chart_output_path}")

    return importance_df


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)

    train_df, test_df, x_train, y_train, x_test, y_test = split_train_test(dataset)

    model = train_random_forest(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    test_result_df = test_df.copy()
    test_result_df["predicted_target"] = y_pred
    test_result_df["predicted_probability"] = y_pred_proba

    prediction_output_path = f"{OUTPUT_FOLDER}/random_forest_predictions.csv"
    test_result_df.to_csv(prediction_output_path, index=False)

    report_text = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    report_output_path = f"{OUTPUT_FOLDER}/random_forest_classification_report.txt"

    with open(report_output_path, "w") as file:
        file.write("RandomForest Classification Report\n")
        file.write("=================================\n\n")
        file.write(f"Train rows: {len(train_df)}\n")
        file.write(f"Test rows: {len(test_df)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report_text)
        file.write("\nConfusion Matrix:\n")
        file.write(str(confusion))

    importance_df = save_feature_importance(model)

    print("\nDay 15 RandomForest model training completed.")
    print("============================================")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix:")
    print(confusion)

    print("\nFeature Importance:")
    print(importance_df)

    print(f"\nSaved predictions to: {prediction_output_path}")
    print(f"Saved report to: {report_output_path}")
    