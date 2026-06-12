import datetime
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats


class DriftDetector:

    def __init__(self, reference_data, threshold=0.05):
        """Initialize drift detector with baseline parameters"""
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history = []

    def detect_drift_ks(self, current_data, feature):
        """Statistical evaluation to spot data divergence"""
        ref_values = self.reference_data[feature]
        curr_values = current_data[feature]
        statistic, pvalue = stats.ks_2samp(ref_values, curr_values)
        drift_detected = pvalue < self.threshold
        return drift_detected, pvalue, statistic

    def check_all_features(self, current_data):
        """Evaluate overall distribution drift across features"""
        results = {}
        drift_detected_any = False

        for feature in self.reference_data.columns:
            drift, p_val, stat = self.detect_drift_ks(current_data, feature)
            results[feature] = {
                "drift_detected": bool(drift),
                "p_value": float(p_val),
                "statistic": float(stat),
            }
            if drift:
                drift_detected_any = True

        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected_any,
            "features": results,
        })
        return drift_detected_any, results

    def save_report(self, filename="drift_report.json"):
        """Export artifacts to filesystem"""
        with open(filename, "w") as f:
            json.dump(self.drift_history, f, indent=2)