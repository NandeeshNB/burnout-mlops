import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, pickle

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Encode categoricals
    le_gender = LabelEncoder()
    le_dept   = LabelEncoder()
    df["gender"]     = le_gender.fit_transform(df["gender"])
    df["department"] = le_dept.fit_transform(df["department"])

    # Features and target
    X = df.drop("burnout_risk", axis=1)
    y = df["burnout_risk"]

    # Scale numerics
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save scaler for inference
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump({"gender": le_gender, "department": le_dept}, f)

    processed = X_scaled.copy()
    processed["burnout_risk"] = y.values
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess("data/raw/healthcare_stress.csv",
               "data/processed/features.csv")