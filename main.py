# main.py

from data_generation.generate_logs import generate_logs
from data_generation.attack_simulation import inject_attacks
from feature_engineering.feature_engineering import run_feature_engineering

from models.isolation_forest import run_isolation_forest
from models.lof import run_lof
from models.one_class_svm import run_one_class_svm
from models.autoencoder import run_autoencoder

from evaluation.evaluation import evaluate_all_models


def main():

    print("\n=== STEP 1: Generating synthetic logs ===")
    generate_logs()

    print("\n=== STEP 2: Injecting attack scenarios ===")
    inject_attacks()

    print("\n=== STEP 3: Feature engineering ===")
    run_feature_engineering()

    print("\n=== STEP 4: Running Isolation Forest ===")
    run_isolation_forest()

    print("\n=== STEP 5: Running LOF ===")
    run_lof()

    print("\n=== STEP 6: Running One-Class SVM ===")
    run_one_class_svm()

    print("\n=== STEP 7: Running Autoencoder ===")
    run_autoencoder()

    print("\n=== STEP 8: Evaluating models ===")
    results = evaluate_all_models()
    print(results.round(4))

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
