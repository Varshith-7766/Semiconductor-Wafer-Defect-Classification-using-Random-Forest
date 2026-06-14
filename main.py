"""
Main Pipeline Script for Semiconductor Wafer Defect Classification.

This script runs the complete end-to-end pipeline:
1. Load and clean the dataset
2. Extract features from wafer maps
3. Prepare data for training
4. Train and compare models
5. Evaluate and visualize results
6. Save all artifacts
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_dataset,
    clean_dataset,
    prepare_features_and_labels,
    ensure_directories,
)
from src.feature_engineering import engineer_features, get_feature_names
from src.train import (
    split_data,
    compare_models,
    evaluate_model,
    save_trained_artifacts,
)
from src.evaluate import (
    print_evaluation_results,
    plot_class_distribution,
    generate_all_plots,
)


def main() -> None:
    start_time = time.time()

    DATA_PATH = "data/LSWMD.pkl"
    MODEL_DIR = "models"
    OUTPUT_DIR = "outputs"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    ensure_directories([MODEL_DIR, OUTPUT_DIR, "data"])

    print("=" * 60)
    print("Semiconductor Wafer Defect Classification")
    print("Using Random Forest with Feature Engineering")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[Step 1/7] Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Columns: {df.columns.tolist()}")

    # Step 2: Clean dataset
    print("\n[Step 2/7] Cleaning dataset...")
    df = clean_dataset(df)

    # Step 3: Feature Engineering
    print("\n[Step 3/7] Extracting features from wafer maps...")
    features_df = engineer_features(df)
    print(f"Features extracted: {features_df.shape[1]} features")

    # Step 4: Prepare data
    print("\n[Step 4/7] Preparing features and labels...")
    X, y, label_encoder, label_mappings = prepare_features_and_labels(
        features_df, df["failureType"]
    )
    plot_class_distribution(df["failureType"], OUTPUT_DIR)

    # Step 5: Train/test split
    print("\n[Step 5/7] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 6: Model comparison
    print("\n[Step 6/7] Comparing model approaches...")
    best_model, best_name, best_acc = compare_models(
        X_train, X_test, y_train, y_test, RANDOM_STATE
    )

    # Step 7: Train final model (larger ensemble with best approach)
    print("\n[Step 7/7] Training final production model...")
    print("\n--- Training Larger Ensemble for Best Performance ---")

    # Use best classifier type with more estimators
    from sklearn.ensemble import RandomForestClassifier

    final_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    print("Final model trained (500 estimators, 21 features).")

    # Evaluate
    print("\n--- Final Model Evaluation ---")
    metrics = evaluate_model(final_model, X_test, y_test, label_encoder)
    print_evaluation_results(metrics, label_encoder)

    # Generate visualizations
    feature_names = get_feature_names()
    generate_all_plots(
        y_test.values,
        metrics["y_pred"],
        final_model,
        feature_names,
        label_encoder,
        OUTPUT_DIR,
    )

    # Save artifacts
    print("\n--- Saving Artifacts ---")
    save_trained_artifacts(final_model, label_encoder, MODEL_DIR)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Best approach: {best_name} (Accuracy: {best_acc:.4f})")
    print(f"Final model accuracy: {metrics['accuracy']:.4f}")
    print(f"Final model F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nArtifacts saved:")
    print(f"  - Model: {MODEL_DIR}/model.pkl")
    print(f"  - Label Encoder: {MODEL_DIR}/label_encoder.pkl")
    print(f"  - Plots: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
