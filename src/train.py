import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)

def train():
    # Always define base_dir first
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use environment variable if set (CI/CD), otherwise use local mlruns
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI (from env): {tracking_uri}")
    else:
        mlruns_dir = os.path.join(base_dir, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        tracking_uri = "file:///" + mlruns_dir.replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI (local): {tracking_uri}")

    # Load params
    params_path = os.path.join(base_dir, "params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)["train"]

    # Load data
    data_path = os.path.join(base_dir, "data", "processed", "features.csv")
    df = pd.read_csv(data_path)
    # ... rest of your code continues unchanged
    X = df.drop("burnout_risk", axis=1)
    y = df["burnout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    mlflow.set_experiment("burnout-risk-prediction")

    # Use lightweight models (low memory usage)
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=200,
            random_state=params["random_state"]
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        ),
    }

    best_f1 = 0
    best_model_name = None
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_param("model_type", name)

            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_proba)

            mlflow.log_metrics({
                "accuracy":  round(acc,  4),
                "f1_score":  round(f1,   4),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "roc_auc":   round(auc,  4),
            })

            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"{name} -> Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_run_id = run.info.run_id

    print(f"\nBest model: {best_model_name} (F1={best_f1:.3f})")

    # Save best model locally
    best_model_uri = f"runs:/{best_run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)

    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {model_path}")

    # Save metrics report
    import json
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    metrics = {
        "model": best_model_name,
        "f1_score": round(best_f1, 4),
        "run_id": best_run_id
    }
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to reports/metrics.json")

if __name__ == "__main__":
    train()