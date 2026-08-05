from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "student_performance.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_score_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.txt"
PLOT_PATH = MODEL_DIR / "prediction_plot.png"

FEATURES = ["study_hours", "sleep_hours", "attendance"]
TARGET = "score"


def load_data() -> pd.DataFrame:
    """Load and validate the project dataset."""
    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURES + [TARGET]) - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data


def train_model() -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    data = load_data()
    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    max_depth=6,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2_score": r2_score(y_test, predictions),
    }

    results = x_test.copy()
    results["actual_score"] = y_test
    results["predicted_score"] = predictions

    return model, metrics, results.sort_values("actual_score")


def save_artifacts(model: Pipeline, metrics: dict[str, float], results: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
        },
        MODEL_PATH,
    )

    METRICS_PATH.write_text(
        "\n".join(
            [
                "Student Performance Predictor Metrics",
                f"Mean Absolute Error: {metrics['mae']:.2f}",
                f"R2 Score: {metrics['r2_score']:.3f}",
            ]
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(results["actual_score"], results["predicted_score"], color="#2563eb", s=60)
    plt.plot([30, 100], [30, 100], color="#f97316", linestyle="--", linewidth=2)
    plt.title("Actual vs Predicted Student Scores")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    model, metrics, results = train_model()
    save_artifacts(model, metrics, results)

    print("Model training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"R2 Score: {metrics['r2_score']:.3f}")


if __name__ == "__main__":
    main()
