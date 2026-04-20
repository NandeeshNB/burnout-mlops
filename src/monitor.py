import pandas as pd
import json, os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def monitor_drift(reference_path, current_path, output_path="reports/drift_report.html"):
    reference = pd.read_csv(reference_path)
    current   = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    print(f"Drift detected: {drift_detected}")
    return drift_detected

if __name__ == "__main__":
    monitor_drift("data/processed/features.csv",
                  "data/processed/features_new.csv")