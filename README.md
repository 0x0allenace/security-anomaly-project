# Behavioral Anomaly Detection within Enterprise Security Event Logs Using Unsupervised Machine Learning

## Overview

This project presents a reproducible framework for behavioral anomaly detection in synthetic enterprise security event logs using unsupervised machine learning. The framework generates controlled synthetic enterprise security events, injects controlled adversarial scenarios, engineers temporal and behavioral features, performs statistical and temporal analysis, and evaluates five complementary anomaly detection models.

The final experimental dataset contains 6,244 security events, consisting of 5,874 normal observations and 370 injected attack observations across four attack scenarios. Ground-truth attack labels are retained for post-hoc evaluation but are not used during model training, preserving the unsupervised learning setting.

**The five evaluated models are:**

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoder
- Gaussian Mixture Model (GMM)

The repository is organized as a modular Python research pipeline covering data generation, attack simulation, feature engineering, statistical analysis, model development, evaluation, and visualization.

## Project Objective

The objective of this project is to design, implement, and evaluate a reproducible behavioral anomaly detection framework capable of identifying suspicious patterns in enterprise security event logs.

The study compares classical, density-based, boundary-based, reconstruction-based, and probabilistic unsupervised learning approaches to examine how different modeling assumptions affect anomaly detection performance.

Particular emphasis is placed on:

- Behavioral deviation from normal user activity
- Temporal activity patterns
- Statistical deviations
- Attack-pattern simulation
- Model comparison under a common feature space
- Threshold-dependent and threshold-independent evaluation

## Project Scope

This project includes:

- Synthetic enterprise security event-log generation
- Controlled adversarial attack simulation
- Temporal and behavioral feature engineering
- Statistical and temporal analysis
- User behavioral baseline analysis
- Composite risk-score construction
- Five unsupervised anomaly detection models
- Centralized quantitative evaluation
- ROC-AUC and PR-AUC analysis
- Confusion-matrix analysis
- Silhouette analysis
- Anomaly-selection sensitivity analysis
- Temporal and feature-space visualization

## Final Experimental Dataset

The final frozen dataset contains:

| Category | Count |
|---|---:|
| Normal events | 5,874 |
| Attack events | 370 |
| Total events | 6,244 |

Injected attack scenarios:

| Attack Scenario | Count |
|---|---:|
| Credential stuffing | 120 |
| Lateral movement | 100 |
| Privilege misuse | 80 |
| Abnormal session duration | 70 |

The dataset is synthetic and does not contain real identities, proprietary telemetry, or confidential enterprise security data.

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
│   ├── autoencoder.py
│   └── gmm.py
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

1. Generate synthetic enterprise security event logs
2. Inject controlled adversarial attack scenarios
3. Engineer temporal, behavioral, and statistical features
4. Perform statistical and temporal analysis
5. Train five unsupervised anomaly detection models
6. Generate continuous anomaly scores and binary anomaly predictions
7. Evaluate models using common quantitative metrics
8. Perform threshold-independent ranking analysis
9. Perform supplementary silhouette analysis
10. Evaluate sensitivity to different anomaly-selection proportions
11. Visualize anomaly behavior across time, users, and feature space

## Simulated Attack Scenarios

Four controlled attack scenarios are injected into the synthetic enterprise environment:

- **Credential stuffing** — repeated authentication failures and abnormal login activity
- **Privilege misuse** — behavioral deviations associated with inappropriate or unusual privilege-related activity
- **Abnormal session duration** — sessions whose duration deviates substantially from typical behavioral patterns
- **Lateral movement** — activity representing movement across enterprise resources and systems

The attack labels are used only for post-hoc evaluation and are excluded from the model feature vector during unsupervised training.

## Feature Engineering

The framework constructs temporal, behavioral, and statistical indicators from the synthetic security events.

### Temporal Features

- Hour of day
- Day-related temporal indicators
- Night-time activity indicator

### Behavioral Features

- Recent login frequency
- Rolling failed-authentication activity
- Session duration
- User behavioral baselines

### Statistical Features

- Session-duration z-score
- Failed-authentication z-score
- Session deviation from baseline

### Composite Feature

- Behavioral composite `risk_score`

The final six-dimensional model feature vector is:

```python
feature_cols = [
    "risk_score",
    "session_duration_min",
    "login_freq_5",
    "failed_attempts_rolling",
    "session_zscore",
    "failed_zscore"
]
```

Attack labels such as `is_attack` and `attack_type` are not included in the model feature vector.

## Statistical & Temporal Analysis

The statistical analysis stage provides contextual understanding of the generated behavioral data before and alongside machine-learning evaluation.

**Methods include:**

- Distribution analysis
- Z-score analysis
- Interquartile Range (IQR) analysis
- Moving-average analysis
- Time-window aggregation
- Seasonal decomposition
- Temporal anomaly-score analysis

**Key observations include:**

- The engineered `risk_score` distribution is positively skewed.
- IQR analysis identifies extreme `risk_score` observations for investigation.
- Session duration and recent login frequency provide useful behavioral signals.
- Moving averages provide a local temporal baseline for short-term deviations.
- Five-minute aggregation provides an alternative representation of behavioral intensity.
- Seasonal decomposition provides limited evidence of strong regular periodic behavior.
- Risk behavior is characterized primarily by episodic spikes rather than consistent seasonal patterns.

## Implemented Models

Five unsupervised anomaly detection models are implemented and evaluated using the same six-feature representation.

### 1. Isolation Forest

Tree-based anomaly detection designed to isolate unusual observations efficiently.

### 2. Local Outlier Factor (LOF)

Density-based method that evaluates observations relative to their local neighborhood.

### 3. One-Class SVM

Boundary-based method that learns a representation of the normal data distribution and identifies observations outside the learned boundary.

### 4. Autoencoder

Neural-network reconstruction approach in which observations with higher reconstruction error can receive higher anomaly scores.

### 5. Gaussian Mixture Model (GMM)

Probabilistic density-estimation approach that models the feature space using Gaussian components. In this experiment, the GMM uses two components with full covariance.

The GMM anomaly score is based on negative log-likelihood, where higher values indicate observations that are less likely under the learned distribution.

## Model Outputs

Each model produces a processed dataset containing its corresponding anomaly score and prediction.

```text
data/processed/processed_logs_with_iforest.csv
data/processed/processed_logs_with_lof.csv
data/processed/processed_logs_with_svm.csv
data/processed/processed_logs_with_autoencoder.csv
data/processed/processed_logs_with_gmm.csv
```

The consolidated comparison results are stored in:

```text
data/processed/model_comparison_results.csv
```

## Evaluation Framework

A centralized evaluation pipeline was developed to provide consistent comparison across all five models.

Evaluation includes:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- True Negatives (TN)
- Precision
- Recall
- F1 Score
- Accuracy
- ROC-AUC
- PR-AUC

Both threshold-dependent and threshold-independent evaluation are used.

### Threshold-Dependent Evaluation

The primary fixed-threshold comparison evaluates model predictions using the common anomaly-selection setting.

### Threshold-Independent Evaluation

ROC-AUC and PR-AUC evaluate the ranking quality of continuous anomaly scores across thresholds.

Because the dataset is imbalanced, PR-AUC is particularly useful for assessing the quality of attack-oriented anomaly ranking.

## Final Model Performance

The final fixed-threshold results are:

| Model | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.4824 | 0.4081 | **0.4422** | 0.9390 |
| GMM | 0.4696 | 0.3973 | 0.4305 | 0.9377 |
| One-Class SVM | 0.2716 | 0.2297 | 0.2489 | 0.9178 |
| Autoencoder | 0.1917 | 0.1622 | 0.1757 | 0.9098 |
| LOF | 0.0671 | 0.0568 | 0.0615 | 0.8973 |

Isolation Forest achieved the highest fixed-threshold precision, recall, F1 score, and accuracy among the evaluated models.

GMM produced the second-highest F1 score and remained competitive with Isolation Forest under the fixed-threshold evaluation.

## Threshold-Independent Performance

The final ROC-AUC and PR-AUC results are:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| GMM | **0.8227** | 0.3712 |
| Isolation Forest | 0.7574 | **0.4608** |
| Autoencoder | 0.7385 | 0.1712 |
| One-Class SVM | 0.6484 | 0.2209 |
| LOF | 0.5622 | 0.1028 |

GMM achieved the highest ROC-AUC (0.8227), indicating the strongest threshold-independent ranking performance according to ROC-AUC among the evaluated models.

Isolation Forest achieved the highest PR-AUC (0.4608), indicating the strongest precision-recall ranking performance among the evaluated models.

These results demonstrate that no single model dominates across every evaluation criterion.

## Key Findings

The final experiment produced several important findings:

- Isolation Forest achieved the highest fixed-threshold F1 score (0.4422).
- Isolation Forest achieved the highest PR-AUC (0.4608).
- GMM achieved the highest ROC-AUC (0.8227).
- GMM achieved the second-highest fixed-threshold F1 score (0.4305).
- Isolation Forest and GMM produced the strongest silhouette scores among the evaluated models.
- One-Class SVM produced intermediate performance but remained below Isolation Forest and GMM across the principal fixed-threshold metrics.
- The Autoencoder produced useful ROC-AUC ranking performance but weaker fixed-threshold F1 and PR-AUC performance.
- LOF produced the weakest performance across the main evaluation criteria.
- Increasing the anomaly-selection proportion generally increased recall, but higher recall did not necessarily result in higher F1.
- No single model dominated every evaluation criterion.

## Confusion-Matrix Analysis

The final confusion matrices were:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Isolation Forest | 5,712 | 162 | 219 | 151 |
| GMM | 5,708 | 166 | 223 | 147 |
| One-Class SVM | 5,646 | 228 | 285 | 85 |
| Autoencoder | 5,621 | 253 | 310 | 60 |
| LOF | 5,582 | 292 | 349 | 21 |

Isolation Forest detected 151 of the 370 injected attack observations at the fixed threshold, while GMM detected 147.

The results also show that all five models produced a substantial number of false negatives, highlighting the difficulty of distinguishing all injected attack behaviors from normal behavioral variation using the selected feature representation.

## Evaluation & Visualization

The project includes quantitative and visual analyses beyond the primary classification metrics.

These include:

- ROC curves using synthetic ground truth
- Precision-recall analysis
- Confusion matrices
- Temporal anomaly-score visualization
- User-based anomaly analysis
- Behavioral feature-space visualization
- Silhouette analysis
- Anomaly-selection sensitivity analysis
- Statistical distribution analysis
- Moving-average temporal analysis
- Time-window aggregation
- Seasonal decomposition

These analyses provide complementary perspectives on model ranking, behavioral separation, temporal patterns, and operating-point sensitivity.

## Silhouette Analysis

Silhouette analysis is used as a supplementary measure of separation between model-generated anomaly and normal groups.

Final silhouette scores:

| Model | Silhouette Score |
|---|---:|
| Isolation Forest | **0.6516** |
| GMM | 0.6502 |
| One-Class SVM | 0.4867 |
| LOF | 0.4221 |
| Autoencoder | 0.3254 |

Isolation Forest and GMM produced the strongest geometric separation according to this supplementary measure.

However, silhouette scores should not be interpreted as direct detection accuracy because the analysis evaluates separation of model-generated groups rather than independently established ground-truth clusters.

## Anomaly-Selection Sensitivity Analysis

Sensitivity analysis evaluates model behavior under anomaly-selection proportions of:

- 1%
- 2%
- 3%
- 5%
- 7%
- 10%

The analysis uses the continuous anomaly scores generated by the canonical experiment and changes the number of observations selected as anomalous. The models are not retrained for each selection proportion.

The analysis shows that:

- Recall generally increases as the selected anomaly proportion increases.
- Isolation Forest reaches its strongest F1 performance at approximately 3% selection.
- GMM reaches its strongest F1 performance around 5%.
- One-Class SVM also shows its strongest F1 performance around 3%.
- Autoencoder F1 increases more gradually as the anomaly-selection proportion increases.
- LOF remains comparatively weak across anomaly-selection settings.
- GMM achieves the highest recall at the 10% selection level.

These findings demonstrate that the proportion of observations selected as anomalous is an important operating parameter and that maximizing recall does not necessarily maximize F1.

## Key Insight

The results suggest that behavioral anomalies are not uniformly separable from normal activity.

Some attack observations overlap substantially with normal behavioral patterns, while others produce stronger statistical or behavioral deviations.

Therefore, anomaly detection performance depends not only on the algorithm but also on:

- Feature representation
- Behavioral context
- Attack characteristics
- Score threshold
- Anomaly-selection proportion
- Evaluation criterion

Rather than treating one algorithm as universally superior, the results support a comparative approach in which different models provide complementary views of anomalous behavior.

## Case Study

The behavioral analysis indicates that anomalous activity can occur as short, event-driven deviations rather than continuous abnormal behavior.

The analysis identified patterns involving:

- Concentrated activity among particular users
- Short-duration behavioral bursts
- Temporal clustering of unusual events
- Deviations in session duration
- Changes in recent authentication activity
- Episodic increases in behavioral risk

These patterns correspond to behavioral characteristics represented in the simulated credential-stuffing, privilege-misuse, abnormal-session, and lateral-movement scenarios.

## Setup

The final experimental environment uses Python 3.11.

Install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the individual pipeline components when reproducing or extending the experiment:

```bash
python3 data_generation/generate_logs.py
python3 data_generation/attack_simulation.py
python3 feature_engineering/feature_engineering.py
python3 models/isolation_forest.py
python3 models/lof.py
python3 models/one_class_svm.py
python3 models/autoencoder.py
python3 models/gmm.py
python3 -m evaluation.evaluation
```

## Reproducibility

The project is designed as a modular and reproducible research pipeline.

The final experiment uses deterministic seeds for synthetic data generation and attack simulation. The canonical dataset contains 6,244 observations and should be treated as the frozen experimental dataset for reproducing the reported manuscript results.

The main orchestration script is:

```bash
python3 main.py
```

`main.py` provides pipeline orchestration for feature processing, model execution, and comparison-result generation.

When modifying the data-generation or attack-simulation configuration, newly generated results should be treated as a new experimental run rather than as the exact canonical results reported in the manuscript.

## Methodology Summary

The experimental methodology follows these stages:

1. Synthetic enterprise security-log generation
2. Controlled attack injection
3. Data preprocessing
4. Temporal and behavioral feature engineering
5. Statistical and temporal analysis
6. Unsupervised model training
7. Continuous anomaly-score generation
8. Fixed-threshold evaluation
9. ROC-AUC and PR-AUC evaluation
10. Silhouette analysis
11. Anomaly-selection sensitivity analysis
12. Comparative interpretation

The attack labels are retained as synthetic ground truth for evaluation but are not supplied to the unsupervised models during training.

## Research Reproducibility

The project separates the major stages of the experiment into modular components:

- `data_generation/` — synthetic event generation and attack injection
- `feature_engineering/` — behavioral and statistical feature construction
- `models/` — anomaly detection algorithms
- `evaluation/` — standardized performance evaluation
- `statistical_analysis/` — statistical and temporal analysis
- `visualization/` — visual analysis and figures
- `outputs/` — generated figures and model-related outputs

This structure is intended to support reproducibility, experimentation, and future extension.

## Conclusion

This project demonstrates that behavioral anomaly detection in security event logs is strongly influenced by the interaction between feature representation, model assumptions, and anomaly-selection thresholds.

Isolation Forest produced the strongest fixed-threshold performance, achieving a precision of 0.4824, recall of 0.4081, F1 score of 0.4422, accuracy of 0.9390, and PR-AUC of 0.4608.

GMM produced the highest ROC-AUC at 0.8227 and a competitive F1 score of 0.4305, demonstrating strong threshold-independent ranking performance according to ROC-AUC.

One-Class SVM provided intermediate performance, while the Autoencoder produced useful ranking performance but weaker fixed-threshold results. LOF produced the weakest performance across the principal evaluation criteria.

The sensitivity analysis further demonstrates that increasing the proportion of observations classified as anomalous generally improves recall but does not necessarily improve F1. This highlights the importance of selecting operating thresholds according to the intended security-monitoring objective.

Overall, the experiment does not identify a universally optimal anomaly detection model. Instead, it demonstrates the value of comparing complementary unsupervised approaches under a common behavioral representation and evaluating them using multiple performance criteria.

## Limitations

The current experiment is intentionally controlled and synthetic. Therefore:

- The dataset does not represent all complexities of real enterprise telemetry.
- Only four attack scenarios are simulated.
- The models use a common six-feature representation.
- Attack labels are used for post-hoc evaluation rather than training.
- The experiment is performed offline rather than in a production streaming environment.
- Results may differ on real-world datasets with different behavioral distributions and attack prevalence.

## Future Work

Potential extensions include:

- Evaluation on real-world enterprise security datasets
- Additional behavioral and resource-access features
- More diverse attack scenarios
- User-resource interaction modeling
- Sequence-based anomaly detection
- Hybrid and ensemble anomaly detection
- Advanced deep-learning architectures
- Online and near-real-time detection
- SOC and SIEM integration
- Automated analyst-oriented alert prioritization

## Tech Stack

- Python 3.11
- Pandas
- NumPy
- Faker
- Scikit-learn
- PyTorch
- Matplotlib
- SciPy
- Statsmodels
- UMAP-learn
- Jupyter

## Repository Status

This repository contains the implementation and supporting materials for the experimental framework described above.

The project is maintained as a research and portfolio artifact for studying behavioral anomaly detection using synthetic enterprise security event logs and unsupervised machine learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Research-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-orange)