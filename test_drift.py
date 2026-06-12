import numpy as np
import pandas as pd
from drift_detector import DriftDetector

# Reference distribution matching original baselines
np.random.seed(42)
n_samples = 1000
reference_data = pd.DataFrame({
    "temperature": np.random.normal(75, 15, n_samples),
    "vibration": np.random.normal(0.5, 0.2, n_samples),
    "pressure": np.random.normal(100, 20, n_samples),
})

detector = DriftDetector(reference_data)

print("Test 1: Evaluating stable patterns (No Drift)")
current_no_drift = pd.DataFrame({
    "temperature": np.random.normal(75, 15, 500),
    "vibration": np.random.normal(0.5, 0.2, 500),
    "pressure": np.random.normal(100, 20, 500),
})
drift_1, results_1 = detector.check_all_features(current_no_drift)
print(f"-> Result: Drift Detected = {drift_1}\n")

print("Test 2: Evaluating operational divergence (Drift Expected)")
current_with_drift = pd.DataFrame({
    "temperature": np.random.normal(
        95, 15, 500
    ),  # Mean shift representing anomalies
    "vibration": np.random.normal(0.8, 0.2, 500),  # Stress anomalies
    "pressure": np.random.normal(100, 20, 500),
})
drift_2, results_2 = detector.check_all_features(current_with_drift)
print(f"-> Result: Drift Detected = {drift_2}")

print("\nDrift Analysis Matrix Summary:")
for feature, res in results_2.items():
    if res["drift_detected"]:
        print(f" [⚠️ DRIFT] {feature:12} | p-value: {res['p_value']:.4f}")

detector.save_report()
print("\n✓ Report written out to: drift_report.json")