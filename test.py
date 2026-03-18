"""
test.py — Test script for the ML Prediction API

Usage:
  1. Run analysis.ipynb to train and save the models.
  2. Start the API:  .venv/bin/uvicorn API:app --host 0.0.0.0 --port 5000
  3. Run this script: python test.py

Results are saved to test_results.csv with columns: A, B, Y, classe
"""
import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:5000"

test_data = [
    {"a": 0.5,  "b": 10},
    {"a": 1.0,  "b": 12},
    {"a": 1.5,  "b": 15},
    {"a": 2.0,  "b": 18},
    {"a": 2.5,  "b": 20},
    {"a": 3.0,  "b": 22},
    {"a": 3.5,  "b": 25},
    {"a": 4.0,  "b": 28},
    {"a": 4.5,  "b": 30},
    {"a": 5.0,  "b": 32},
    {"a": 5.5,  "b": 35},
    {"a": 6.0,  "b": 38},
    {"a": 0.2,  "b": 11},
    {"a": 3.2,  "b": 24},
    {"a": 5.8,  "b": 37},
]

results = []

for item in test_data:
    # 1. /predict — get predicted Y from A and B
    res_p = requests.post(f"{BASE_URL}/predict", params=item).json()
    pred_y = res_p["Y"]

    # 2. /classe — get cluster class from A, B and predicted Y
    res_c = requests.get(f"{BASE_URL}/classe", params={"a": item["a"], "b": item["b"], "y": pred_y}).json()

    results.append({"A": item["a"], "B": item["b"], "Y": pred_y, "classe": res_c["classe"]})

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv('test_results.csv', index=False)
print("Results saved to test_results.csv")
