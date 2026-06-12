# Production-Grade Predictive Maintenance System (MLOps Lifecycle)

Welcome to the ultimate finale of the ML Engineering journey. This project showcases the complete end-to-end MLOps deployment pipeline for industrial equipment predictive maintenance. Over the course of Weeks 13 to 16, this repository evolved from a local model exploration workspace into a fully automated, self-healing, cloud-monitored, production REST API.

---

## 🏗️ System Architecture Overview

This project implements a closed-loop MLOps system that balances real-time model serving with automated statistical governance.

[Image of MLOps pipeline architecture showing data engineering model training API deployment drift detection and automated retraining loop]

1. **Production Gateway**: FastAPI serves failure predictions under high-throughput constraints.
2. **Telemetry Tracking**: Real-time server diagnostics are collected locally via a polling infrastructure.
3. **Statistical Governance**: incoming feature distributions are monitored against training baselines to detect drift.
4. **Self-Healing Retraining Loop**: Drif detection triggers automated cloud training pipelines to push fresh model weights back to the registry.

---

## 📅 Week-by-Week Breakdown & Deliverables

### 🧪 Weeks 13 & 14: Model Tracking, Versioning & Rollback Strategy
* **Experiment Tracking**: Integrated **MLflow** to track hyperparameter configurations, metrics, and training run schemas.
* **Model Registry**: Saved and versioned trained weights directly under the `PredictiveMaintenance` model namespace.
* **Transformation Pipeline**: Serialized feature normalization weights into `scaler.pkl` to prevent data leakage during deployment.
* **Rollback Auditing**: Implemented a fallback validation framework simulating model stage movements, ensuring that the system can gracefully rollback to Version 1 (`Production` stage) if secondary updates fail evaluation.

### 🌐 Week 15: Production REST API & Live Metrics Telemetry
* **Microservice Framework**: Wrapped the production MLflow model inside a lightweight, highly responsive **FastAPI** web server application (`api/main.py`).
* **Data Validation**: Structured input schemas using Pydantic V2 to enforce data types for features (`temperature`, `vibration`, `pressure`, `rpm`, `age_days`).
* **Telemetry Diagnostics (`/metrics`)**: Designed localized metrics tracking to capture runtime data payload trends, including:
  * Total request counters.
  * Anomalies/Failure predictions captured by the model.
  * Real-time rolling prediction latency (in milliseconds).
* **Live Dashboarding (`monitor_dashboard.py`)**: Built a separate script that pulls diagnostic logs every 5 seconds to provide an active system health overview.
* **Stress Load Testing (`load_test.py`)**: Validated system stability by throwing 100 concurrent requests at the prediction endpoints.

### 🔄 Week 16: Automated Drift Detection & CI/CD Pipelines
* **Statistical Drift Monitoring (`drift_detector.py`)**: Implemented a statistical detector utilizing the **Kolmogorov-Smirnov test (KS-Test)**. It actively evaluates current data arrays against baseline distributions, writing results to `drift_report.json`.
* **Autonomous Retraining Pipeline (`retrain_pipeline.py`)**: Engineered a self-healing script that automatically triggers when significant distribution deviations are caught. It ingests new parameters, fits an updated **XGBoost Classifier**, and registers Version 3 directly to MLflow.
* **Continuous Integration (`.github/workflows/test.yml`)**: Configured a **GitHub Actions** script that listens for pushes to the `main` branch, spins up a virtual Linux server, builds dependencies, and runs automated `pytest` unit tests.
* **Scheduled Operations (`.github/workflows/retrain.yml`)**: Set up a weekly CRON tab scheduler to automatically monitor drift trends, execute retraining runs, and preserve historical drift artifact logs.

---

## 📂 Project Directory Structure

```text
Real-World-ML-Engineering/
├── .github/
│   └── workflows/
│       ├── test.yml                 # Automated testing CI pipeline
│       └── retrain.yml              # Scheduled retraining workflow
├── api/
│   └── main.py                      # Core FastAPI prediction server
├── tests/
│   └── test_model.py                # Automated pytest validation test suites
├── drift_detector.py                # Kolmogorov-Smirnov statistical logic
├── test_drift.py                    # Evaluator script for data distribution shifts
├── retrain_pipeline.py              # Automated XGBoost retraining engine
├── monitor_dashboard.py             # Active telemetry visualization suite
├── test_api.py                      # Endpoint functionality matrix tester
├── load_test.py                     # 100-request automated stress load engine
├── scaler.pkl                       # Feature normalization pipeline weights
├── requirements.txt                 # Complete project dependency manifest
└── README.md                        # Documentation handbook
