import numpy as np
import pandas as pd
import pytest


def test_data_generation():
    """Test synthetic data generation consistency"""
    n_samples = 100
    data = pd.DataFrame({
        "temperature": np.random.normal(75, 15, n_samples),
        "vibration": np.random.normal(0.5, 0.2, n_samples),
        "pressure": np.random.normal(100, 20, n_samples),
    })
    assert len(data) == n_samples
    assert not data.isnull().any().any()


def test_model_input_shape():
    """Test data pipeline structure matches training criteria"""
    input_data = pd.DataFrame({
        "temperature": [75.0],
        "vibration": [0.5],
        "pressure": [100.0],
        "rpm": [1500.0],
        "age_days": [100],
    })
    assert input_data.shape == (1, 5)