"""
training_data.py
Generates synthetic normal and anomalous GraphQL feature vectors for training.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "max_depth",         # Max nesting depth of query
    "total_fields",      # Total number of fields selected
    "unique_fields",     # Unique field names
    "alias_count",       # Number of aliases used
    "introspection_count",  # Count of __schema / __type / __typename
    "fragment_count",    # Number of inline/named fragments
    "estimated_cost",    # Estimated resolver cost (fields * depth weighting)
    "payload_size",      # Raw query string length in bytes
    "field_entropy",     # Shannon entropy of field name distribution
    "nesting_variance",  # Variance of nesting depth across selection sets
]


def _entropy(values):
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    p = arr / arr.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def generate_normal_samples(n=2000, seed=42):
    """Realistic normal GraphQL query features."""
    rng = np.random.default_rng(seed)
    samples = {
        "max_depth":           rng.integers(1, 6, n),
        "total_fields":        rng.integers(1, 20, n),
        "unique_fields":       rng.integers(1, 18, n),
        "alias_count":         rng.integers(0, 3, n),
        "introspection_count": rng.integers(0, 1, n),
        "fragment_count":      rng.integers(0, 2, n),
        "estimated_cost":      rng.integers(1, 50, n),
        "payload_size":        rng.integers(20, 300, n),
        "field_entropy":       rng.uniform(0.5, 2.5, n),
        "nesting_variance":    rng.uniform(0.0, 1.5, n),
    }
    df = pd.DataFrame(samples)
    df["label"] = 0  # 0 = normal
    return df


def generate_anomalous_samples(n=500, seed=99):
    """Known-bad GraphQL query features covering all 6 threat types."""
    rng = np.random.default_rng(seed)
    frames = []

    # --- Deep nesting / abnormal depth ---
    deep = {
        "max_depth":           rng.integers(12, 25, n // 6),
        "total_fields":        rng.integers(5, 20, n // 6),
        "unique_fields":       rng.integers(5, 18, n // 6),
        "alias_count":         rng.integers(0, 2, n // 6),
        "introspection_count": np.zeros(n // 6, int),
        "fragment_count":      rng.integers(0, 1, n // 6),
        "estimated_cost":      rng.integers(80, 200, n // 6),
        "payload_size":        rng.integers(200, 600, n // 6),
        "field_entropy":       rng.uniform(0.5, 2.0, n // 6),
        "nesting_variance":    rng.uniform(2.0, 8.0, n // 6),
    }
    frames.append(pd.DataFrame(deep))

    # --- Field explosion ---
    explosion = {
        "max_depth":           rng.integers(1, 5, n // 6),
        "total_fields":        rng.integers(80, 200, n // 6),
        "unique_fields":       rng.integers(60, 180, n // 6),
        "alias_count":         rng.integers(0, 3, n // 6),
        "introspection_count": np.zeros(n // 6, int),
        "fragment_count":      rng.integers(0, 2, n // 6),
        "estimated_cost":      rng.integers(150, 400, n // 6),
        "payload_size":        rng.integers(500, 2000, n // 6),
        "field_entropy":       rng.uniform(3.0, 6.0, n // 6),
        "nesting_variance":    rng.uniform(0.0, 1.5, n // 6),
    }
    frames.append(pd.DataFrame(explosion))

    # --- Alias abuse ---
    alias = {
        "max_depth":           rng.integers(1, 4, n // 6),
        "total_fields":        rng.integers(20, 60, n // 6),
        "unique_fields":       rng.integers(1, 5, n // 6),
        "alias_count":         rng.integers(15, 50, n // 6),
        "introspection_count": np.zeros(n // 6, int),
        "fragment_count":      rng.integers(0, 2, n // 6),
        "estimated_cost":      rng.integers(100, 300, n // 6),
        "payload_size":        rng.integers(300, 1000, n // 6),
        "field_entropy":       rng.uniform(0.1, 1.0, n // 6),
        "nesting_variance":    rng.uniform(0.0, 0.5, n // 6),
    }
    frames.append(pd.DataFrame(alias))

    # --- Introspection abuse ---
    intro = {
        "max_depth":           rng.integers(3, 8, n // 6),
        "total_fields":        rng.integers(10, 40, n // 6),
        "unique_fields":       rng.integers(5, 30, n // 6),
        "alias_count":         rng.integers(0, 2, n // 6),
        "introspection_count": rng.integers(5, 20, n // 6),
        "fragment_count":      rng.integers(0, 1, n // 6),
        "estimated_cost":      rng.integers(50, 150, n // 6),
        "payload_size":        rng.integers(100, 500, n // 6),
        "field_entropy":       rng.uniform(1.0, 3.0, n // 6),
        "nesting_variance":    rng.uniform(0.5, 3.0, n // 6),
    }
    frames.append(pd.DataFrame(intro))

    # --- High resolver cost ---
    cost = {
        "max_depth":           rng.integers(4, 8, n // 6),
        "total_fields":        rng.integers(30, 80, n // 6),
        "unique_fields":       rng.integers(20, 70, n // 6),
        "alias_count":         rng.integers(0, 5, n // 6),
        "introspection_count": np.zeros(n // 6, int),
        "fragment_count":      rng.integers(0, 3, n // 6),
        "estimated_cost":      rng.integers(300, 1000, n // 6),
        "payload_size":        rng.integers(400, 1500, n // 6),
        "field_entropy":       rng.uniform(2.0, 5.0, n // 6),
        "nesting_variance":    rng.uniform(1.0, 4.0, n // 6),
    }
    frames.append(pd.DataFrame(cost))

    # --- Data exfiltration (high entropy, many unique sensitive-looking fields) ---
    exfil = {
        "max_depth":           rng.integers(2, 6, n // 6),
        "total_fields":        rng.integers(15, 50, n // 6),
        "unique_fields":       rng.integers(14, 48, n // 6),
        "alias_count":         rng.integers(0, 3, n // 6),
        "introspection_count": np.zeros(n // 6, int),
        "fragment_count":      rng.integers(0, 2, n // 6),
        "estimated_cost":      rng.integers(80, 250, n // 6),
        "payload_size":        rng.integers(300, 900, n // 6),
        "field_entropy":       rng.uniform(5.0, 8.0, n // 6),
        "nesting_variance":    rng.uniform(0.5, 2.5, n // 6),
    }
    frames.append(pd.DataFrame(exfil))

    df = pd.concat(frames, ignore_index=True)
    df["label"] = 1  # 1 = anomalous
    return df


def get_full_dataset(seed=42):
    normal = generate_normal_samples(2000, seed)
    anomalous = generate_anomalous_samples(500, seed)
    df = pd.concat([normal, anomalous], ignore_index=True).sample(frac=1, random_state=seed)
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df["label"].values
    return X, y, FEATURE_COLUMNS
