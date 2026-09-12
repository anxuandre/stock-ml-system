import os
import json
import joblib

from src.data_loader import download_data
from src.features import add_features
from src.labels import add_labels
from src.config import LABEL_HORIZON, TRAIN_SPLIT_DATE
from src.train_lr import train_and_evaluate_lr
from src.train_rf import train_and_evaluate_rf

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d",
]


def prepare_dataset():
    df = download_data()
    df = add_features(df)
    df = add_labels(df)
    # Track the observation used by each target before dropping incomplete rows.
    df["label_end_date"] = df["Date"].shift(-LABEL_HORIZON)
    df = df.dropna().copy()

    train_df = df[
        (df["Date"] < TRAIN_SPLIT_DATE)
        & (df["label_end_date"] < TRAIN_SPLIT_DATE)
    ].copy()
    test_df = df[df["Date"] >= TRAIN_SPLIT_DATE].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training and test partitions must both contain complete rows after "
            "feature warmup and label-horizon purging. Check the data date range "
            "and TRAIN_SPLIT_DATE."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test


def save_model(model, path):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)


def save_best_model_metadata(model_name, model_path, feature_columns):
    metadata = {
        "model_name": model_name,
        "model_path": model_path,
        "feature_columns": feature_columns,
    }

    with open("models/best_model.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_dataset()

    lr_model, lr_metrics = train_and_evaluate_lr(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_test, y_test)

    print("\nLogistic Regression metrics:")
    for k, v in lr_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nRandom Forest metrics:")
    for k, v in rf_metrics.items():
        print(f"{k}: {v:.4f}")

    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_model = rf_model
        best_name = "random_forest"
    else:
        best_model = lr_model
        best_name = "logistic_regression"

    model_path = f"models/{best_name}.joblib"
    save_model(best_model, model_path)
    save_best_model_metadata(best_name, model_path, FEATURE_COLUMNS)

    print(f"\nSaved best model: {best_name}")
    print("Saved metadata: models/best_model.json")
