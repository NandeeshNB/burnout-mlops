# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "age": np.random.randint(24, 60, n),
    "gender": np.random.choice(["Male","Female","Other"], n),
    "department": np.random.choice(["ICU","ER","General","Surgery","Pediatrics"], n),
    "weekly_hours": np.random.randint(40, 80, n),
    "sleep_hours": np.round(np.random.uniform(4, 8, n), 1),
    "physical_activity": np.random.randint(0, 7, n),
    "job_satisfaction": np.random.randint(1, 10, n),
    "workload_score": np.random.randint(1, 10, n),
    "support_from_management": np.random.randint(1, 10, n),
    "years_experience": np.random.randint(1, 30, n),
    "on_call_frequency": np.random.randint(0, 15, n),
})
df["burnout_risk"] = ((df["weekly_hours"] > 60) &
                      (df["job_satisfaction"] < 5) &
                      (df["sleep_hours"] < 6)).astype(int)

df.to_csv("data/raw/healthcare_stress.csv", index=False)
print("Dataset created:", df.shape)