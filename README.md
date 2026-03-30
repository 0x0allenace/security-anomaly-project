
# Behavioral Anomaly Detection in Security Event Logs Using Advanced Unsupervised Machine Learning Models

## Overview

This project builds a high-fidelity synthetic enterprise security dataset and applies behavioral anomaly detection techniques using unsupervised machine learning models. The pipeline simulates realistic user activity, injects adversarial attack scenarios, engineers behavioral features, and evaluates anomaly detection models in a controlled environment.


## Project Objective

The goal of this project is to design and evaluate a framework capable of detecting anomalous behavioral patterns in enterprise security logs. The study focuses on comparing classical and deep learning anomaly detection approaches within a controlled synthetic environment.


## Current Scope

The project currently includes:

- Synthetic enterprise security log generation  
- Adversarial attack injection  
- Behavioral feature engineering  
- Exploration notebook for validation, visualization, and model evaluation  
- Implementation of Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, and Autoencoder  

Planned stages include:

- Comparative evaluation of models  
- Final report documentation  

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

1. Generate baseline synthetic security logs  
2. Inject attack scenarios into baseline logs  
3. Engineer temporal, behavioral, and statistical features  
4. Apply statistical and temporal analysis techniques  
5. Inspect and validate outputs in notebook
6. Apply unsupervised anomaly detection models:
	- 	Isolation Forest
	- 	Local Outlier Factor (LOF)
	- 	One-Class SVM
	- 	Autoencoder    

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

## Key Insights
	- 	Isolation Forest performs strongly in detecting global outliers.
	- 	LOF shows limited effectiveness due to weak local density structure in the dataset.
	- 	One-Class SVM provides moderate performance but struggles with overlapping feature boundaries.
	- 	Autoencoder achieves the best performance by learning complex behavioral patterns and identifying anomalies through reconstruction error.

## Current Progress

	-	Full synthetic data pipeline completed
	-	Feature engineering pipeline established
	-	Statistical and temporal analysis implemented
	-	All four anomaly detection models implemented
	-	Model outputs, visualizations, and evaluations completed
	-	Ready for final comparative analysis and reporting 


## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate baseline logs:
```bash
python3 data_generation/generate_logs.py
```

Inject attacks:
```bash
python3 data_generation/attack_simulation.py
```

Run feature engineering pipeline::
```bash
python3 feature_engineering/feature_engineering.py
```

Run Isolation Forest:
```bash
python3 models/isolation_forest.py
```

Run Local Outlier Factor (LOF):
```bash
python3 models/lof.py
```

Run One-Class SVM:
```bash
python3 models/one_class_svm.py
```
Run Autoencoder:
```bash
python3 models/autoencoder.py
```

## Research Notes

This project is being developed incrementally with aligned code, notebook inspection, and report methodology updates to maintain reproducibility and research integrity.

## Sample Output

Example attack distribution:

- Normal: 5874
- Credential Stuffing: 120
- Lateral Movement: 100
- Privilege Misuse: 80
- Abnormal Session: 70

## Methodology

This project follows a structured anomaly detection pipeline:

1. Synthetic log generation simulating enterprise user activity
2. Injection of adversarial attack behaviors (credential abuse, lateral movement, etc.)
3. Behavioral feature engineering using temporal, statistical, and user-based deviations
4. Application of unsupervised anomaly detection models
5. Evaluation using statistical and model-based metrics

This approach ensures reproducibility, interpretability, and controlled experimentation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Active-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-orange)