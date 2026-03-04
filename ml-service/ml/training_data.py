"""
training_data.py
Generates synthetic normal and anomalous GraphQL feature vectors for training.
Now also supports corpus-based generation from real query strings.
"""

import numpy as np
import pandas as pd
from typing import Optional

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


def generate_normal_from_corpus(
    n: Optional[int] = None,
    noise_factor: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Extract real feature vectors from the GraphQL query corpus and return
    as a labeled DataFrame identical in schema to generate_normal_samples().

    Args:
        n:            If given, augment with slight noise to reach exactly n
                      samples (useful for balancing). If None, returns one
                      row per corpus query.
        noise_factor: Std deviation of Gaussian noise added during augment
                      (as fraction of each feature value). Default 0.05 = 5%.
        seed:         Random seed for reproducibility.

    Returns:
        DataFrame with FEATURE_COLUMNS + label=0 column.
    """
    try:
        from ml.query_corpus import NORMAL_QUERIES
        from ml.query_feature_extractor import extract_features_batch
    except ImportError:
        # Fallback: couldn't import corpus — return empty DataFrame
        print("[training_data] WARNING: Could not import query corpus. "
              "Returning empty corpus DataFrame.")
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["label"])

    # Extract features from every corpus query
    raw = extract_features_batch(NORMAL_QUERIES)
    corpus_df = pd.DataFrame(raw, columns=FEATURE_COLUMNS)
    corpus_df["label"] = 0

    # If n is not specified or is satisfied by the corpus, just return
    if n is None or len(corpus_df) >= n:
        result = corpus_df.head(n) if n else corpus_df
        return result.reset_index(drop=True)

    # --- Augmentation to reach exactly n samples ---
    rng = np.random.default_rng(seed)
    rows_needed = n - len(corpus_df)
    base = corpus_df[FEATURE_COLUMNS].values.astype(float)

    # Sample with replacement from base, then add small Gaussian noise
    idxs = rng.integers(0, len(base), size=rows_needed)
    noise = rng.normal(0.0, noise_factor, size=(rows_needed, len(FEATURE_COLUMNS)))
    augmented = base[idxs] * (1.0 + noise)
    augmented = np.clip(augmented, 0, None)  # keep non-negative

    aug_df = pd.DataFrame(augmented, columns=FEATURE_COLUMNS)
    aug_df["label"] = 0

    combined = pd.concat([corpus_df, aug_df], ignore_index=True)
    return combined.reset_index(drop=True)


def generate_blended_normal(
    n_total: int = 3000,
    corpus_ratio: float = 0.60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Blend corpus-derived normal samples (corpus_ratio) with synthetic samples
    (1 - corpus_ratio) to create a richer training distribution.

    Args:
        n_total:      Total number of normal samples desired.
        corpus_ratio: Fraction derived from real corpus queries. Default 0.60.
        seed:         Random seed.

    Returns:
        Shuffled DataFrame with FEATURE_COLUMNS + label=0.
    """
    n_corpus = int(n_total * corpus_ratio)
    n_synth  = n_total - n_corpus

    corpus_df = generate_normal_from_corpus(n=n_corpus, seed=seed)
    synth_df  = generate_normal_samples(n=n_synth, seed=seed + 1)

    combined = pd.concat([corpus_df, synth_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    corpus_size = len(corpus_df)
    print(f"[training_data] Blended normal set: "
          f"{corpus_size} corpus ({corpus_ratio:.0%}) + "
          f"{n_synth} synthetic ({1-corpus_ratio:.0%}) = {len(combined)} total")
    return combined
