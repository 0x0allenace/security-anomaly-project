# config.py

# =====================
# DATA PATHS
# =====================
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

IFOREST_OUTPUT = "data/processed/processed_logs_with_iforest.csv"
LOF_OUTPUT = "data/processed/processed_logs_with_lof.csv"
SVM_OUTPUT = "data/processed/processed_logs_with_svm.csv"
AUTOENCODER_OUTPUT = "data/processed/processed_logs_with_autoencoder.csv"

MODEL_COMPARISON_OUTPUT = "data/processed/model_comparison_results.csv"


# =====================
# FEATURES
# =====================
FEATURE_COLUMNS = [
    "risk_score",
    "session_duration_min",
    "login_freq_5",
    "failed_attempts_rolling",
    "session_zscore",
    "failed_zscore"
]


# =====================
# MODEL PARAMETERS
# =====================

# Isolation Forest
IFOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.05,
    "random_state": 42
}

# LOF
LOF_PARAMS = {
    "n_neighbors": 20,
    "contamination": 0.05
}

# One-Class SVM
SVM_PARAMS = {
    "kernel": "rbf",
    "gamma": "scale",
    "nu": 0.05
}

# Autoencoder
AUTOENCODER_PARAMS = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}
