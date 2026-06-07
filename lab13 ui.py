import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost  # XGBoost native logging ke liye
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ==================== MLflow Setup ====================
mlflow.set_experiment('predictive-maintenance')
mlflow.set_tracking_uri('http://localhost:5000')

print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
print(f'Experiment name: {mlflow.get_experiment_by_name("predictive-maintenance").name}\n')

# ==================== Load Data ====================
data = pd.read_csv('artificial_maintenance_data.csv')

print(f'✅ Artificial data loaded! Shape: {data.shape}\n')

# ==================== Data Preprocessing ====================
X = data.drop('failure', axis=1)
y = data['failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Training set: {X_train.shape}')
print(f'Test set: {X_test.shape}\n')

# ==================== Task 3.1: Logistic Regression ====================
with mlflow.start_run(run_name='logistic_regression'):
    C = 1.0
    max_iter = 1000
    mlflow.log_param('C', C)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('model_type', 'LogisticRegression')

    model_lr = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model_lr.fit(X_train_scaled, y_train)

    y_pred = model_lr.predict(X_test_scaled)
    y_pred_proba = model_lr.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))
    mlflow.sklearn.log_model(model_lr, 'model')

# ==================== Task 3.2: Random Forest ====================
with mlflow.start_run(run_name='random_forest'):
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('min_samples_split', min_samples_split)
    mlflow.log_param('model_type', 'RandomForest')

    model_rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, random_state=42
    )
    model_rf.fit(X_train_scaled, y_train)

    y_pred = model_rf.predict(X_test_scaled)
    y_pred_proba = model_rf.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))
    mlflow.sklearn.log_model(model_rf, 'model')

# ==================== Task 3.3: XGBoost  ====================
with mlflow.start_run(run_name='xgboost') as run:
    n_estimators = 100
    max_depth = 6
    learning_rate = 0.1

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('model_type', 'XGBoost')

    model_xgb = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42, eval_metric='logloss'
    )
    model_xgb.fit(X_train_scaled, y_train)

    y_pred = model_xgb.predict(X_test_scaled)
    y_pred_proba = model_xgb.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))

    # Native XGBoost logging use ki
    mlflow.xgboost.log_model(model_xgb, 'model')

    # FIX: Scaler ko local file mein dump karke MLflow Artifacts mein save kiya
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact('scaler.pkl', artifact_path='model_artifacts')
    os.remove('scaler.pkl')  # Local file clear kar di

    print(f'✅ XGBoost Model and Scaler logged to MLflow successfully!')

print('✅ All three models trained and logged successfully!\n')


import pickle

# Scaler ko save kar diya taaki doosri file isko use kar sake
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler file 'scaler.pkl' successfully save ho gayi hai!")