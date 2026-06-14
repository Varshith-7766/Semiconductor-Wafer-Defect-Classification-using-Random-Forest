"""
Evaluation and Visualization Module for Semiconductor Wafer Defect Classification.

This module generates evaluation metrics, confusion matrices, and feature importance plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List


def print_evaluation_results(metrics: Dict[str, Any], label_encoder: LabelEncoder) -> None:
    """
    Print all evaluation metrics to console.

    Args:
        metrics: Dictionary containing evaluation metrics.
        label_encoder: Fitted LabelEncoder for class names.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("=" * 60)


def plot_class_distribution(
    labels: pd.Series,
    output_dir: str = "outputs",
    label_encoder: LabelEncoder = None,
) -> None:
    """
    Plot and save the class distribution.

    Args:
        labels: Series of label strings.
        output_dir: Directory to save the plot.
        label_encoder: Optional fitted LabelEncoder.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    class_counts = labels.value_counts()

    colors = sns.color_palette("husl", len(class_counts))
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors)

    plt.xticks(
        range(len(class_counts)),
        class_counts.index,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.xlabel("Defect Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Wafer Defect Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    filepath = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class distribution plot saved to {filepath}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: str = "outputs",
) -> None:
    """
    Plot and save the confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        label_encoder: Fitted LabelEncoder.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {filepath}")


def plot_feature_importance(
    model,
    feature_names: List[str],
    output_dir: str = "outputs",
) -> None:
    """
    Plot and save the full feature importance chart.

    Args:
        model: Trained classifier with feature_importances_ attribute.
        feature_names: List of feature names.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(importances)), importances[indices], color="steelblue", edgecolor="navy", alpha=0.8)
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("Feature Importance (All Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {filepath}")


def plot_top_features(
    model,
    feature_names: List[str],
    top_n: int = 20,
    output_dir: str = "outputs",
) -> None:
    """
    Plot and save the top-N feature importance chart.

    Args:
        model: Trained classifier with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 7))
    colors = sns.color_palette("viridis", len(indices))
    bars = plt.barh(
        range(len(indices)),
        importances[indices][::-1],
        color=colors,
        edgecolor="darkgreen",
        alpha=0.85,
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices][::-1],
        fontsize=10,
    )
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(f"Top-{top_n} Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "feature_importance_top20.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top-{top_n} feature importance plot saved to {filepath}")


def generate_all_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    feature_names: List[str],
    label_encoder: LabelEncoder,
    output_dir: str = "outputs",
) -> None:
    """
    Generate all visualization plots.

    Args:
        y_test: True test labels.
        y_pred: Predicted test labels.
        model: Trained classifier.
        feature_names: List of feature names.
        label_encoder: Fitted LabelEncoder.
        output_dir: Directory to save plots.
    """
    print("\n--- Generating Visualizations ---")

    plot_confusion_matrix(y_test, y_pred, label_encoder, output_dir)
    plot_feature_importance(model, feature_names, output_dir)
    plot_top_features(model, feature_names, top_n=20, output_dir=output_dir)

    print("--- All visualizations generated ---\n")


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
