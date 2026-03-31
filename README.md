# Behavioral Anomaly Detection in Security Event Logs Using Advanced Unsupervised Machine Learning Models

## Overview

This project presents a comprehensive framework for behavioral anomaly detection in enterprise security logs using advanced unsupervised machine learning techniques. The system generates a high-fidelity synthetic dataset, simulates realistic attack scenarios, engineers behavioral features, and evaluates multiple anomaly detection models in a controlled and reproducible environment.

## Project Objective

The objective of this project is to design, implement, and evaluate a robust anomaly detection framework capable of identifying suspicious behavioral patterns in enterprise security logs. The study compares classical and deep learning-based unsupervised models to determine their effectiveness in detecting complex and context-dependent anomalies.

## Project Scope

This project includes:

- Synthetic enterprise security log generation  
- Adversarial attack simulation  
- Behavioral feature engineering  
- Statistical and temporal analysis  
- Implementation of multiple anomaly detection models  
- Centralized evaluation pipeline  
- Advanced visualization and interpretability analysis  

## Project Structure

```text
security-anomaly-project/
├── data/
│   ├── raw/
│   └── processed/
├── data_generation/
│   ├── generate_logs.py
│   └── attack_simulation.py
├── feature_engineering/
│   └── feature_engineering.py
├── statistical_analysis/
│   └── stats_analysis.py
├── models/
│   ├── isolation_forest.py
│   ├── lof.py
│   ├── one_class_svm.py
│   └── autoencoder.py
├── evaluation/
│   ├── metrics.py
│   └── evaluation.py
├── visualization/
│   └── visualize.py
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── figures/
│   └── models/
├── report/
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

## Implemented Pipeline
	1.	Generate synthetic enterprise security logs
	2.	Inject adversarial attack scenarios
	3.	Engineer temporal, behavioral, and statistical features
	4.	Perform statistical and temporal analysis
	5.	Train unsupervised anomaly detection models
	6.	Evaluate model performance using standardized metrics
	7.	Visualize anomaly patterns across time, users, and feature space 

## Simulated Attack Scenarios

- Credential stuffing  
- Privilege misuse  
- Abnormal session duration  
- Lateral movement

## Feature Engineering

- Time-based features  
- Rolling login behavior  
- Statistical deviation metrics  
- User baseline deviation  
- Composite risk score  
- Moving average smoothing (temporal trends)  
- Time-window aggregation (behavior summarization)  
  

## Statistical & Temporal Analysis

- Distribution analysis (histograms)  
- Z-score based anomaly detection  
- Interquartile Range (IQR) method  
- Moving averages for temporal smoothing  
- Time-window aggregation for burst detection  
- Seasonal decomposition for trend and residual analysis 

## Implemented Models

	The following anomaly detection models have been implemented and evaluated:
	- 	Isolation Forest (global anomaly detection)
	- 	Local Outlier Factor (LOF) (density-based local anomaly detection)
	- 	One-Class SVM (boundary-based anomaly detection)
	- 	Autoencoder (deep learning-based reconstruction method)

## Evaluation Framework

A centralized evaluation pipeline was developed to ensure consistent performance measurement across all models. Metrics include:

- True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
- Precision, Recall, F1 Score, Accuracy
- ROC-like curves and AUC

Results are stored in:
```test
data/processed/model_comparison_results.csv
```

## Key Findings

	•	The Autoencoder achieved the best overall performance, demonstrating strong capability in detecting both global and subtle anomalies.
	•	Isolation Forest performed well as a classical baseline, particularly for global outliers.
	•	One-Class SVM showed moderate effectiveness but struggled with overlapping feature boundaries.
	•	LOF underperformed due to weak local density structure in the dataset.

## Visualization & Insights

Advanced visualizations were used to interpret model behavior:
- ROC-like curves → score-based model comparison
- Temporal heatmaps → anomaly patterns across time
- User-based heatmaps → identification of high-risk users
- t-SNE projection → feature space structure and anomaly separability


## Key Insight

Anomalies in the dataset are best characterized as:

“Locally deviating within structured behavioral manifolds rather than globally separable clusters”

## Case Study

User-level analysis revealed that anomalous behavior is:
- Concentrated among specific users
- Occurring in short, event-driven bursts
- Distributed across time rather than continuous

This reflects realistic enterprise attack patterns such as credential misuse and lateral movement.


## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```
Run individual components:
```bash
python3 data_generation/generate_logs.py
python3 data_generation/attack_simulation.py
python3 feature_engineering/feature_engineering.py

python3 models/isolation_forest.py
python3 models/lof.py
python3 models/one_class_svm.py
python3 models/autoencoder.py

python3 -m evaluation.evaluation
```
## Reproducibility

The project supports full pipeline execution via:
```bash
python3 main.py
```
This regenerates data, retrains models, and recomputes evaluation results.

## Methodology Summary

This project follows a structured anomaly detection approach:

- Synthetic log generation
- Attack injection
- Behavioral feature engineering
- Unsupervised model training
- Quantitative evaluation
- Qualitative visualization and interpretation

## Conclusion

This project demonstrates that effective behavioral anomaly detection requires a combination of robust feature engineering and models capable of capturing complex, non-linear patterns. Reconstruction-based methods, particularly Autoencoders, are highly effective in identifying subtle anomalies within structured behavioral data.

## Tech Stack
	•	Python 3.10+
	•	Scikit-learn
	•	PyTorch / TensorFlow (Autoencoder)
	•	Pandas, NumPy
	•	Matplotlib, Seaborn

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Active-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-orange)