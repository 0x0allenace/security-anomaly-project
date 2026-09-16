import pandas as pd

from data_generation.generate_logs import main as generate_logs
from data_generation.attack_simulation import main as inject_attacks
from feature_engineering.feature_engineering import run_feature_engineering

from models.isolation_forest import (
    run_isolation_forest,
    get_default_iforest_features,
)

from models.lof import run_lof

from models.one_class_svm import run_one_class_svm

from models.autoencoder import (
    run_autoencoder,
    get_default_autoencoder_features,
)

from models.gmm import (
    run_gmm,
    get_default_gmm_features,
)

from evaluation.evaluation import evaluate_all_models


def main():

    # ---------------------------------------------------------
    # STEP 1: Generate synthetic normal logs
    # ---------------------------------------------------------

    print("\n=== STEP 1: Generating synthetic logs ===")

    generate_logs()

    # ---------------------------------------------------------
    # STEP 2: Inject attack scenarios
    # ---------------------------------------------------------

    print("\n=== STEP 2: Injecting attack scenarios ===")

    inject_attacks()

    # ---------------------------------------------------------
    # STEP 3: Feature engineering
    # ---------------------------------------------------------

    print("\n=== STEP 3: Feature engineering ===")

    input_path = "data/raw/logs_with_attacks.csv"
    processed_path = "data/processed/processed_logs.csv"

    run_feature_engineering(
        input_path=input_path,
        output_path=processed_path
    )

    # ---------------------------------------------------------
    # Load engineered dataset
    # ---------------------------------------------------------

    df = pd.read_csv(processed_path)

    print(f"\nLoaded engineered dataset: {len(df)} rows")

    # ---------------------------------------------------------
    # Common feature set
    # ---------------------------------------------------------

    feature_cols = [
        "risk_score",
        "session_duration_min",
        "login_freq_5",
        "failed_attempts_rolling",
        "session_zscore",
        "failed_zscore"
    ]

    # ---------------------------------------------------------
    # STEP 4: Isolation Forest
    # ---------------------------------------------------------

    print("\n=== STEP 4: Running Isolation Forest ===")

    iforest_features = get_default_iforest_features()

    df_iforest, iforest_model = run_isolation_forest(
        df=df,
        feature_cols=iforest_features,
        contamination=0.05,
        random_state=42
    )

    iforest_output = (
        "data/processed/processed_logs_with_iforest.csv"
    )

    df_iforest.to_csv(
        iforest_output,
        index=False
    )

    print(
        f"Isolation Forest results saved to: "
        f"{iforest_output}"
    )

    # ---------------------------------------------------------
    # STEP 5: LOF
    # ---------------------------------------------------------

    print("\n=== STEP 5: Running LOF ===")

    df_lof = run_lof(
        df=df,
        feature_cols=feature_cols,
        n_neighbors=20,
        contamination=0.05
    )

    lof_output = (
        "data/processed/processed_logs_with_lof.csv"
    )

    df_lof.to_csv(
        lof_output,
        index=False
    )

    print(
        f"LOF results saved to: "
        f"{lof_output}"
    )

    # ---------------------------------------------------------
    # STEP 6: One-Class SVM
    # ---------------------------------------------------------

    print("\n=== STEP 6: Running One-Class SVM ===")

    df_svm = run_one_class_svm(
        df=df,
        feature_cols=feature_cols,
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )

    svm_output = (
        "data/processed/processed_logs_with_svm.csv"
    )

    df_svm.to_csv(
        svm_output,
        index=False
    )

    print(
        f"One-Class SVM results saved to: "
        f"{svm_output}"
    )

    # ---------------------------------------------------------
    # STEP 7: Autoencoder
    # ---------------------------------------------------------

    print("\n=== STEP 7: Running Autoencoder ===")

    autoencoder_features = get_default_autoencoder_features()

    (
        df_autoencoder,
        autoencoder_model,
        autoencoder_scaler,
        autoencoder_threshold,
    ) = run_autoencoder(
        df=df,
        feature_cols=autoencoder_features,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        contamination=0.05,
        random_state=42
    )

    autoencoder_output = (
        "data/processed/processed_logs_with_autoencoder.csv"
    )

    df_autoencoder.to_csv(
        autoencoder_output,
        index=False
    )

    print(
        f"Autoencoder results saved to: "
        f"{autoencoder_output}"
    )

    print(
        f"Autoencoder anomaly threshold: "
        f"{autoencoder_threshold:.6f}"
    )

    # ---------------------------------------------------------
    # STEP 8: Gaussian Mixture Model (GMM)
    # ---------------------------------------------------------

    print("\n=== STEP 8: Running GMM ===")

    gmm_features = get_default_gmm_features()

    (
        df_gmm,
        gmm_model,
        gmm_scaler,
        gmm_threshold,
    ) = run_gmm(
        df=df,
        feature_cols=gmm_features,
        n_components=2,
        covariance_type="full",
        contamination=0.05,
        random_state=42
    )

    gmm_output = (
        "data/processed/processed_logs_with_gmm.csv"
    )

    df_gmm.to_csv(
        gmm_output,
        index=False
    )

    print(
        f"GMM results saved to: "
        f"{gmm_output}"
    )

    print(
        f"GMM anomaly threshold: "
        f"{gmm_threshold:.6f}"
    )

    # ---------------------------------------------------------
    # STEP 9: Evaluate all models
    # ---------------------------------------------------------

    print("\n=== STEP 9: Evaluating models ===")

    results = evaluate_all_models()

    print("\nModel Evaluation Results:")

    print(
        results.round(4)
    )

    # ---------------------------------------------------------
    # Save evaluation results
    # ---------------------------------------------------------

    evaluation_output = (
        "data/processed/model_comparison_results.csv"
    )

    results.to_csv(
        evaluation_output,
        index=False
    )

    print(
        f"\nEvaluation results saved to: "
        f"{evaluation_output}"
    )

    # ---------------------------------------------------------
    # Pipeline complete
    # ---------------------------------------------------------

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
