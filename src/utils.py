"""
Utility functions for data loading, preprocessing, and pipeline management.
"""

import os
import io
import pickle
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder


class _OldPandasUnpickler(pickle._Unpickler):
    """
    Custom unpickler that handles old pandas pickle files (pre-1.0).
    This version (0.23.x era) used 'pandas.indexes' and 'pandas._new_Index'.
    """

    _CLASS_MAP = {
        ("pandas.indexes.base", "Index"): pd.Index,
        ("pandas.indexes.numeric", "Int64Index"): pd.Index,
        ("pandas.indexes.numeric", "Float64Index"): pd.Index,
        ("pandas.indexes.range", "RangeIndex"): pd.RangeIndex,
        ("pandas.indexes.category", "CategoricalIndex"): pd.Index,
        ("pandas.indexes.multi", "MultiIndex"): pd.MultiIndex,
        ("pandas.core.indexes.numeric", "Int64Index"): pd.Index,
        ("pandas.core.indexes.numeric", "UInt64Index"): pd.Index,
        ("pandas.core.indexes.numeric", "Float64Index"): pd.Index,
        ("pandas.core.indexes.range", "RangeIndex"): pd.RangeIndex,
        ("pandas.core.indexes.category", "CategoricalIndex"): pd.Index,
        ("pandas.core.indexes.multi", "MultiIndex"): pd.MultiIndex,
    }

    _MODULE_MAP = {
        "pandas.indexes": "pandas.core.indexes",
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.numeric": "pandas.core.indexes.numeric",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.category": "pandas.core.indexes.category",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
    }

    def find_class(self, module: str, name: str):
        # Check class-level redirects first
        key = (module, name)
        if key in self._CLASS_MAP:
            return self._CLASS_MAP[key]

        # Check module-level redirects
        if module in self._MODULE_MAP:
            module = self._MODULE_MAP[module]

        return super().find_class(module, name)

    def load_reduce(self):
        stack = self.stack
        args = stack.pop()
        func = stack[-1]

        # Handle pandas._new_Index which was used in old pandas pickles
        # It takes (IndexClass, data_dict) and creates an Index
        if hasattr(func, "__module__") and func.__module__ == "pandas" and func.__name__ == "_new_Index":
            try:
                cls = args[0] if isinstance(args, (tuple, list)) else args
                # Try to construct the Index
                if isinstance(cls, type) and issubclass(cls, pd.Index):
                    stack[-1] = cls.__new__(cls)
                else:
                    stack[-1] = func(*args)
            except Exception:
                stack[-1] = pd.Index([])
            return

        try:
            stack[-1] = func(*args)
        except TypeError:
            # Try alternative construction methods
            if isinstance(args, (tuple, list)) and len(args) >= 1:
                if isinstance(args[0], type) and issubclass(args[0], pd.Index):
                    stack[-1] = args[0].__new__(args[0])
                    return
            raise


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the WM-811K wafer dataset from pickle file.

    Args:
        data_path: Path to the LSWMD.pkl file.

    Returns:
        Loaded pandas DataFrame.
    """
    # First, try the standard pandas reader
    try:
        df = pd.read_pickle(data_path)
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        return df
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Standard load failed ({e}), trying compatibility loader...")

    # Try the custom unpickler
    try:
        with open(data_path, "rb") as f:
            df = _OldPandasUnpickler(f, encoding="latin1").load()
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing samples with missing, empty, or unknown failure labels.

    Args:
        df: Raw DataFrame from load_dataset.

    Returns:
        Cleaned DataFrame.
    """
    initial_size = len(df)

    # Ensure failureType column exists
    if "failureType" not in df.columns:
        raise ValueError("Column 'failureType' not found in dataset.")

    # Remove rows where failureType is missing (None, NaN, or empty)
    df = df.dropna(subset=["failureType"])

    # Remove rows where failureType is an empty list or array
    mask = df["failureType"].apply(lambda x: _is_valid_label(x))
    df = df[mask].copy()

    # Flatten failureType to string labels
    df["failureType"] = df["failureType"].apply(_flatten_label)

    # Remove unknown or empty labels
    df = df[df["failureType"].str.strip() != ""]
    df = df[df["failureType"].str.lower() != "unknown"]

    # Reset index
    df = df.reset_index(drop=True)

    cleaned_size = len(df)
    removed = initial_size - cleaned_size
    print(f"Dataset cleaned: {removed} samples removed, {cleaned_size} samples remaining")

    return df


def _is_valid_label(label) -> bool:
    """Check if a label is valid (not None, not empty)."""
    if label is None:
        return False
    if isinstance(label, (list, np.ndarray)):
        return len(label) > 0
    if isinstance(label, str):
        return len(label.strip()) > 0
    return True


def _flatten_label(label) -> str:
    """Flatten a label to a string."""
    if isinstance(label, (list, np.ndarray)):
        if len(label) > 0:
            if isinstance(label[0], (list, np.ndarray)):
                return str(label[0][0]).strip()
            return str(label[0]).strip()
    return str(label).strip()


def prepare_features_and_labels(
    features_df: pd.DataFrame, labels: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, dict]:
    """
    Prepare features and labels for training.

    Args:
        features_df: Engineered feature DataFrame.
        labels: Target label Series.

    Returns:
        Tuple of (features DataFrame, encoded labels, label encoder, label mappings).
    """
    # Handle NaN values in features
    features_df = features_df.fillna(0.0)

    # Replace infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0.0)

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Create label mappings
    label_mappings = {
        idx: label for idx, label in enumerate(le.classes_)
    }

    print(f"Features shape: {features_df.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}")

    return features_df, pd.Series(encoded_labels, index=features_df.index), le, label_mappings


def save_model(model, filepath: str) -> None:
    """Save a trained model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_label_encoder(le: LabelEncoder, filepath: str) -> None:
    """Save the label encoder to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to {filepath}")


def load_label_encoder(filepath: str) -> LabelEncoder:
    """Load the label encoder from disk."""
    with open(filepath, "rb") as f:
        le = pickle.load(f)
    print(f"Label encoder loaded from {filepath}")
    return le


def ensure_directories(directories: List[str]) -> None:
    """Ensure that all required directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    # Test utilities
    df = load_dataset("data/LSWMD.pkl")
    print(df.head())
    print(df.columns.tolist())
