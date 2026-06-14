# Semiconductor Wafer Defect Classification using Random Forest

A complete end-to-end Machine Learning project for classifying semiconductor wafer defect patterns using engineered statistical features and optimized Random Forest classifier.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Random Forest Architecture](#random-forest-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Future Improvements](#future-improvements)

---

## Introduction

Semiconductor manufacturing is a highly complex process where wafer defects can significantly impact yield and product quality. Identifying and classifying these defects is crucial for:

- **Yield optimization**: Understanding defect patterns helps improve manufacturing processes
- **Quality control**: Early detection prevents defective chips from reaching customers
- **Root cause analysis**: Different defect patterns indicate different manufacturing issues

This project uses machine learning to automatically classify wafer defect patterns from the WM-811K dataset, achieving high classification accuracy through engineered features and optimized Random Forest models.

## Dataset Overview

The **WM-811K** (Wafer Map) dataset contains:

- **811,457** wafer map samples
- Each sample is a 2D map representing the pass/fail status of individual dies on a wafer
- **9 labeled defect patterns**: Center, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, Triangle, None
- **30x30** pixel resolution per wafer map

### Defect Types

| Label | Description |
|-------|-------------|
| Center | Defect concentrated in the center of the wafer |
| Edge-Loc | Defects localized near the wafer edge |
| Edge-Ring | Defects forming a ring pattern at the edge |
| Loc | Localized defect cluster |
| Near-full | Defects covering nearly the entire wafer |
| Random | Randomly distributed defects |
| Scratch | Linear scratch-like defect pattern |
| Triangle | Triangular defect pattern |
| No defect | No defects detected |

## Project Structure

```
Semiconductor-Wafer-Defect-Classification/
│
├── data/
│   └── LSWMD.pkl                  # WM-811K wafer dataset
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis notebook
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── feature_engineering.py      # Feature extraction from wafer maps
│   ├── train.py                    # Model training and hyperparameter tuning
│   ├── evaluate.py                 # Evaluation metrics and visualizations
│   └── utils.py                    # Data loading and utility functions
│
├── models/
│   ├── model.pkl                   # Trained Random Forest model
│   └── label_encoder.pkl           # Fitted LabelEncoder
│
├── outputs/
│   ├── class_distribution.png      # Class distribution plot
│   ├── confusion_matrix.png        # Confusion matrix heatmap
│   ├── feature_importance.png      # Full feature importance chart
│   └── feature_importance_top20.png # Top-20 feature importance
│
├── main.py                         # Main pipeline script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Semiconductor-Wafer-Defect-Classification.git
cd Semiconductor-Wafer-Defect-Classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the WM-811K dataset
2. Extract 21 statistical and spatial features from each wafer map
3. Train and compare multiple Random Forest variants
4. Perform hyperparameter tuning with GridSearchCV
5. Generate evaluation metrics and visualizations
6. Save the trained model and all artifacts

### Running EDA

Open the Jupyter notebook for exploratory data analysis:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Data Preprocessing

### Cleaning Steps

1. **Remove missing labels**: Samples where `failureType` is None, NaN, or empty
2. **Remove unknown labels**: Samples with unrecognized defect patterns
3. **Flatten label format**: Convert nested list/array labels to simple strings
4. **Reset indices**: Ensure clean sequential indexing

### Label Encoding

Defect type strings are encoded to numerical labels using `sklearn.preprocessing.LabelEncoder`:

```python
LabelEncoder().fit_transform(["Center", "Edge-Loc", "Edge-Ring", ...])
# Output: [0, 1, 2, ...]
```

## Feature Engineering

### Overview

We extract **21 numerical features** from each wafer map across three categories:

### 1. Basic Features (6 features)

| Feature | Description |
|---------|-------------|
| `wafer_height` | Height of the wafer map (rows) |
| `wafer_width` | Width of the wafer map (columns) |
| `total_dies` | Total number of dies (height × width) |
| `failed_dies` | Count of defective dies |
| `normal_dies` | Count of non-defective dies |
| `failure_ratio` | Proportion of failed dies |

### 2. Statistical Features (8 features)

| Feature | Description |
|---------|-------------|
| `row_failure_mean` | Mean failure density across rows |
| `row_failure_std` | Standard deviation of row failure density |
| `row_failure_max` | Maximum row failure density |
| `row_failure_min` | Minimum row failure density |
| `col_failure_mean` | Mean failure density across columns |
| `col_failure_std` | Standard deviation of column failure density |
| `col_failure_max` | Maximum column failure density |
| `col_failure_min` | Minimum column failure density |

### 3. Spatial Features (7 features)

| Feature | Description |
|---------|-------------|
| `center_failure_density` | Failure density in the wafer center region |
| `edge_failure_density` | Failure density along wafer edges |
| `center_to_edge_ratio` | Ratio of center to edge failure density |
| `failure_entropy` | Shannon entropy of failure distribution |
| `failure_concentration_score` | How concentrated failures are in dense regions |
| `failure_cluster_count` | Number of distinct failure clusters |
| `largest_cluster_size` | Size of the largest connected failure cluster |

## Random Forest Architecture

### Base Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

### Class Imbalance Handling

We compare three approaches:

1. **class_weight="balanced"**: Automatic weight adjustment based on class frequencies
2. **BalancedRandomForestClassifier**: Undersamples majority class during training
3. **SMOTE + Random Forest**: Synthetic oversampling of minority classes

The best-performing approach is selected for final model training.

## Hyperparameter Tuning

### GridSearchCV Configuration

- **Cross-validation**: 5-fold StratifiedKFold
- **Scoring metric**: Accuracy
- **Parallel execution**: n_jobs=-1

### Parameter Grid

| Parameter | Values |
|-----------|--------|
| `n_estimators` | [100, 200, 300] |
| `max_depth` | [None, 20, 30, 40] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |
| `max_features` | ["sqrt", "log2"] |
| `class_weight` | ["balanced", "balanced_subsample"] |

Total combinations: 3 × 4 × 3 × 3 × 2 × 2 = **432** configurations

## Evaluation Metrics

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision (weighted)** | Average precision weighted by class support |
| **Recall (weighted)** | Average recall weighted by class support |
| **F1 Score (weighted)** | Harmonic mean of precision and recall |

### Visualizations

1. **Class Distribution Plot**: Shows the distribution of defect types in the dataset
2. **Confusion Matrix Heatmap**: Detailed prediction analysis across all classes
3. **Feature Importance Chart**: Full ranking of all 21 features
4. **Top-20 Feature Importance**: Focused view of most important features

## Results

### Performance Summary

The optimized Random Forest model achieves strong performance on the cleaned WM-811K dataset:

| Metric | Score |
|--------|-------|
| Accuracy | ~90%+ |
| Precision | ~90%+ |
| Recall | ~90%+ |
| F1 Score | ~90%+ |

*Exact values depend on the specific train/test split and hyperparameter tuning results.*

### Model Comparison

| Approach | Description | Relative Performance |
|----------|-------------|---------------------|
| Balanced RF | Built-in class balancing | Good |
| SMOTE + RF | Synthetic oversampling | Good |
| GridSearchCV RF | Tuned hyperparameters | Best |

## Feature Importance

### Key Findings

The most discriminative features for wafer defect classification are typically:

1. **failure_ratio**: Overall defect percentage is highly informative
2. **failure_cluster_count**: Number of distinct defect clusters
3. **largest_cluster_size**: Size of the dominant defect region
4. **center_failure_density**: Spatial distribution in center
5. **row_failure_std / col_failure_std**: Variability in failure patterns

### Interpretation

- **Spatial features** capture the geometric patterns that distinguish defect types
- **Statistical features** quantify the distribution characteristics
- **Basic features** provide overall defect magnitude information

## Future Improvements

### Potential Enhancements

1. **Deep Learning Approaches**
   - CNN-based classification using raw wafer map images
   - Transfer learning from pre-trained image models
   - Autoencoders for unsupervised feature learning

2. **Advanced Feature Engineering**
   - Texture features (GLCM, LBP)
   - Fourier transform features
   - Wavelet-based multi-resolution analysis

3. **Ensemble Methods**
   - XGBoost, LightGBM, CatBoost comparisons
   - Stacking ensemble with multiple base learners
   - Voting classifier with diverse models

4. **Data Augmentation**
   - Rotation, flipping, and scaling augmentations
   - Synthetic minority oversampling optimization
   - Domain-specific augmentation techniques

5. **Deployment**
   - REST API for real-time inference
   - Docker containerization
   - Model monitoring and drift detection

### Known Limitations

- Feature engineering may not capture all spatial relationships
- Class imbalance can affect minority class performance
- 30x30 resolution limits fine-grained pattern detection

## License

This project is open source and available under the MIT License.

## Acknowledgments

- WM-811K dataset from [Kaggle](https://www.kaggle.com/qingyi/wm811k-wafer-map)
- Scikit-learn and imbalanced-learn communities
- Semiconductor manufacturing research community
