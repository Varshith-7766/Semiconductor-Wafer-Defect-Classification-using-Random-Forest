"""
Feature Engineering Module for Semiconductor Wafer Defect Classification.

This module extracts statistical, spatial, and structural features from wafer maps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from scipy.stats import entropy


def extract_basic_features(wafer_map: np.ndarray) -> Dict[str, float]:
    """
    Extract basic structural features from a wafer map.

    Args:
        wafer_map: 2D numpy array representing the wafer map.

    Returns:
        Dictionary containing basic features.
    """
    if wafer_map.size == 0:
        return _empty_basic_features()

    height, width = wafer_map.shape
    total_dies = height * width
    failed_dies = int(np.sum(wafer_map == 1))
    normal_dies = int(np.sum(wafer_map == 0))
    failure_ratio = failed_dies / total_dies if total_dies > 0 else 0.0

    return {
        "wafer_height": float(height),
        "wafer_width": float(width),
        "total_dies": float(total_dies),
        "failed_dies": float(failed_dies),
        "normal_dies": float(normal_dies),
        "failure_ratio": failure_ratio,
    }


def extract_statistical_features(wafer_map: np.ndarray) -> Dict[str, float]:
    """
    Extract row-wise and column-wise failure density statistics.

    Args:
        wafer_map: 2D numpy array representing the wafer map.

    Returns:
        Dictionary containing statistical features.
    """
    if wafer_map.size == 0:
        return _empty_statistical_features()

    # Row-wise failure density
    row_failure = np.mean(wafer_map == 1, axis=1)
    row_stats = {
        "row_failure_mean": float(np.mean(row_failure)),
        "row_failure_std": float(np.std(row_failure)),
        "row_failure_max": float(np.max(row_failure)),
        "row_failure_min": float(np.min(row_failure)),
    }

    # Column-wise failure density
    col_failure = np.mean(wafer_map == 1, axis=0)
    col_stats = {
        "col_failure_mean": float(np.mean(col_failure)),
        "col_failure_std": float(np.std(col_failure)),
        "col_failure_max": float(np.max(col_failure)),
        "col_failure_min": float(np.min(col_failure)),
    }

    return {**row_stats, **col_stats}


def extract_spatial_features(wafer_map: np.ndarray) -> Dict[str, float]:
    """
    Extract spatial and structural features from a wafer map.

    Args:
        wafer_map: 2D numpy array representing the wafer map.

    Returns:
        Dictionary containing spatial features.
    """
    if wafer_map.size == 0:
        return _empty_spatial_features()

    height, width = wafer_map.shape
    failure_mask = (wafer_map == 1).astype(np.int32)

    # Center failure density
    center_h, center_w = height // 2, width // 2
    margin_h = max(1, height // 4)
    margin_w = max(1, width // 4)
    center_region = failure_mask[
        center_h - margin_h : center_h + margin_h,
        center_w - margin_w : center_w + margin_w,
    ]
    center_failure_density = float(np.mean(center_region))

    # Edge failure density
    edge_mask = np.zeros_like(failure_mask)
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1
    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_failure_density = float(np.mean(failure_mask[edge_mask == 1]))

    # Center-to-edge failure ratio
    center_to_edge_ratio = (
        center_failure_density / edge_failure_density
        if edge_failure_density > 0
        else 0.0
    )

    # Failure distribution entropy
    failure_row_dist = np.mean(failure_mask, axis=1)
    failure_col_dist = np.mean(failure_mask, axis=0)

    row_entropy = entropy(failure_row_dist + 1e-10)
    col_entropy = entropy(failure_col_dist + 1e-10)
    failure_entropy = float((row_entropy + col_entropy) / 2)

    # Failure concentration score (ratio of failures in densest region)
    total_failures = np.sum(failure_mask)
    if total_failures > 0:
        kernel = np.ones((3, 3))
        conv = ndimage.convolve(failure_mask.astype(float), kernel)
        max_concentration = float(np.max(conv))
        concentration_score = max_concentration / total_failures
    else:
        concentration_score = 0.0

    # Failure cluster count and largest cluster size
    labeled_array, num_clusters = ndimage.label(failure_mask)
    if num_clusters > 0:
        cluster_sizes = ndimage.sum(failure_mask, labeled_array, range(1, num_clusters + 1))
        largest_cluster_size = float(np.max(cluster_sizes))
    else:
        largest_cluster_size = 0.0

    return {
        "center_failure_density": center_failure_density,
        "edge_failure_density": edge_failure_density,
        "center_to_edge_ratio": center_to_edge_ratio,
        "failure_entropy": failure_entropy,
        "failure_concentration_score": float(concentration_score),
        "failure_cluster_count": float(num_clusters),
        "largest_cluster_size": largest_cluster_size,
    }


def extract_all_features(wafer_map: np.ndarray) -> Dict[str, float]:
    """
    Extract all features from a single wafer map.

    Args:
        wafer_map: 2D numpy array representing the wafer map.

    Returns:
        Dictionary containing all extracted features.
    """
    basic = extract_basic_features(wafer_map)
    statistical = extract_statistical_features(wafer_map)
    spatial = extract_spatial_features(wafer_map)
    return {**basic, **statistical, **spatial}


def _empty_basic_features() -> Dict[str, float]:
    """Return empty basic features for edge cases."""
    return {
        "wafer_height": 0.0,
        "wafer_width": 0.0,
        "total_dies": 0.0,
        "failed_dies": 0.0,
        "normal_dies": 0.0,
        "failure_ratio": 0.0,
    }


def _empty_statistical_features() -> Dict[str, float]:
    """Return empty statistical features for edge cases."""
    return {
        "row_failure_mean": 0.0,
        "row_failure_std": 0.0,
        "row_failure_max": 0.0,
        "row_failure_min": 0.0,
        "col_failure_mean": 0.0,
        "col_failure_std": 0.0,
        "col_failure_max": 0.0,
        "col_failure_min": 0.0,
    }


def _empty_spatial_features() -> Dict[str, float]:
    """Return empty spatial features for edge cases."""
    return {
        "center_failure_density": 0.0,
        "edge_failure_density": 0.0,
        "center_to_edge_ratio": 0.0,
        "failure_entropy": 0.0,
        "failure_concentration_score": 0.0,
        "failure_cluster_count": 0.0,
        "largest_cluster_size": 0.0,
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the entire dataframe.

    Args:
        df: DataFrame with 'waferMap' column containing numpy arrays.

    Returns:
        DataFrame with engineered features.
    """
    features_list = []
    for idx, row in df.iterrows():
        wafer_map = row["waferMap"]
        features = extract_all_features(wafer_map)
        features_list.append(features)

    features_df = pd.DataFrame(features_list, index=df.index)
    return features_df


def get_feature_names() -> List[str]:
    """Return the list of all feature names."""
    basic = list(_empty_basic_features().keys())
    statistical = list(_empty_statistical_features().keys())
    spatial = list(_empty_spatial_features().keys())
    return basic + statistical + spatial


if __name__ == "__main__":
    print("Feature names:", get_feature_names())
    print("Number of features:", len(get_feature_names()))
