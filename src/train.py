"""
Training Module for Semiconductor Wafer Defect Classification.

This module handles model training, hyperparameter tuning, and model persistence.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

from .utils import save_model, save_label_encoder


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified train-test split.

    Args:
        X: Feature matrix.
        y: Target labels.
        test_size: Proportion of test data.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a baseline Random Forest model with balanced class weights.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Trained RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Baseline model trained.")
    return model


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Best estimator from randomized search.
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


def train_balanced_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> BalancedRandomForestClassifier:
    """
    Train a Balanced Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Trained BalancedRandomForestClassifier.
    """
    model = BalancedRandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Balanced Random Forest model trained.")
    return model


def train_with_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest model after SMOTE resampling.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Trained RandomForestClassifier.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE resampled: {X_resampled.shape[0]} samples")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_resampled, y_resampled)
    print("SMOTE + Random Forest model trained.")
    return model


def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, str, float]:
    """
    Compare different approaches and select the best one.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        random_state: Random seed.

    Returns:
        Tuple of (best model, model name, best accuracy).
    """
    models = {}

    # Baseline with class_weight="balanced"
    print("\n--- Training Baseline (class_weight=balanced) ---")
    baseline = train_baseline_model(X_train, y_train, random_state)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    models["baseline_balanced"] = (baseline, baseline_acc)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Balanced Random Forest
    print("\n--- Training Balanced Random Forest ---")
    brf = train_balanced_random_forest(X_train, y_train, random_state)
    brf_acc = accuracy_score(y_test, brf.predict(X_test))
    models["balanced_rf"] = (brf, brf_acc)
    print(f"Balanced RF accuracy: {brf_acc:.4f}")

    # SMOTE + Random Forest
    print("\n--- Training SMOTE + Random Forest ---")
    smote_rf = train_with_smote(X_train, y_train, random_state)
    smote_acc = accuracy_score(y_test, smote_rf.predict(X_test))
    models["smote_rf"] = (smote_rf, smote_acc)
    print(f"SMOTE + RF accuracy: {smote_acc:.4f}")

    # Select best model
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_acc = models[best_name]

    print(f"\n=== Best Model: {best_name} (Accuracy: {best_acc:.4f}) ===")

    return best_model, best_name, best_acc


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train the final model with tuned hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Trained RandomForestClassifier.
    """
    print("\n--- Hyperparameter Tuning with GridSearchCV ---")
    best_model = hyperparameter_tuning(X_train, y_train, random_state)
    return best_model


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.
        label_encoder: Fitted LabelEncoder.

    Returns:
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

    return metrics


def save_trained_artifacts(
    model,
    label_encoder: LabelEncoder,
    model_dir: str = "models",
) -> None:
    """
    Save the trained model and label encoder.

    Args:
        model: Trained classifier.
        label_encoder: Fitted LabelEncoder.
        model_dir: Directory to save artifacts.
    """
    import os
    os.makedirs(model_dir, exist_ok=True)

    save_model(model, os.path.join(model_dir, "model.pkl"))
    save_label_encoder(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))


if __name__ == "__main__":
    print("Training module loaded successfully.")
