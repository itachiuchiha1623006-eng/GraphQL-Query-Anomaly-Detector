"""
attack_generator.py
====================
Generates labeled feature vectors for 16 GraphQL attack categories.

Each attack function returns a list of dicts with:
  - All 10 ML features (matching FEATURE_COLUMNS)
  - 'label'       : 1 (anomalous)
  - 'attack_type' : human-readable category
  - 'severity'    : 'low' | 'medium' | 'high' | 'critical'

Attack categories:
  DoS:
    1.  deep_nesting            – Abnormal query depth
    2.  recursive_fragments     – Fragment recursion amplification
    3.  resolver_cost_explosion – High resolver invocation cost
    4.  massive_field_selection – Field count explosion
    5.  huge_argument_lists     – Payload-size DoS
    6.  batch_query_abuse       – Batched operation amplification

  Access / Auth:
    7.  alias_explosion         – Alias-based DoS / result amplification
    8.  idor_field_scanning     – IDOR via field enumeration
    9.  broken_auth_brute       – Batching attack for brute force

  Injection:
    10. sql_nosql_injection      – Injection via argument field entropy
    11. os_command_injection     – Command-like payload patterns
    12. ssrf_crlf_injection      – SSRF/CRLF via large unusual payloads

  Recon / Information Disclosure:
    13. introspection_abuse      – Schema discovery
    14. mutation_spam            – Mutation flooding
    15. insecure_default_config  – Exploiting permissive defaults

  Frequency:
    16. query_frequency_spike    – Rate anomaly (simulated feature vector)
"""

import numpy as np
from ml.training_data import FEATURE_COLUMNS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample(rng, n, **kwargs):
    """Build n samples from kwargs where each value is a (low, high) tuple."""
    rows = []
    for _ in range(n):
        row = {k: float(rng.integers(v[0], v[1] + 1) if isinstance(v, tuple) and all(isinstance(x, int) for x in v)
               else rng.uniform(v[0], v[1]) if isinstance(v, tuple) else float(v))
               for k, v in kwargs.items()}
        rows.append(row)
    return rows


def _make(samples, attack_type, severity):
    out = []
    for s in samples:
        row = {col: s.get(col, 0.0) for col in FEATURE_COLUMNS}
        row['label']       = 1
        row['attack_type'] = attack_type
        row['severity']    = severity
        out.append(row)
    return out


# ── 1. Deep Nesting ──────────────────────────────────────────────────────────
def deep_nesting(n=300, seed=1):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (12, 30),
        total_fields        = (10, 30),
        unique_fields       = (8, 25),
        alias_count         = (0, 2),
        introspection_count = 0,
        fragment_count      = (0, 1),
        estimated_cost      = (200, 900),
        payload_size        = (200, 800),
        field_entropy       = (0.5, 2.0),
        nesting_variance    = (4.0, 12.0),
    )
    return _make(samples, 'deep_nesting', 'critical')


# ── 2. Recursive Fragments ───────────────────────────────────────────────────
def recursive_fragments(n=200, seed=2):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (3, 7),
        total_fields        = (50, 200),
        unique_fields       = (5, 15),
        alias_count         = (3, 9),
        introspection_count = 0,
        fragment_count      = (5, 20),
        estimated_cost      = (500, 2000),
        payload_size        = (800, 4000),
        field_entropy       = (0.5, 1.5),
        nesting_variance    = (1.0, 3.0),
    )
    return _make(samples, 'recursive_fragments', 'critical')


# ── 3. Resolver Cost Explosion ───────────────────────────────────────────────
def resolver_cost_explosion(n=200, seed=3):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (5, 8),
        total_fields        = (40, 100),
        unique_fields       = (30, 90),
        alias_count         = (0, 5),
        introspection_count = 0,
        fragment_count      = (0, 3),
        estimated_cost      = (600, 2000),
        payload_size        = (400, 1500),
        field_entropy       = (2.0, 5.0),
        nesting_variance    = (2.0, 6.0),
    )
    return _make(samples, 'resolver_cost_explosion', 'high')


# ── 4. Massive Field Selection ───────────────────────────────────────────────
def massive_field_selection(n=200, seed=4):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 4),
        total_fields        = (100, 500),
        unique_fields       = (80, 450),
        alias_count         = (0, 3),
        introspection_count = 0,
        fragment_count      = (0, 2),
        estimated_cost      = (300, 1500),
        payload_size        = (1000, 5000),
        field_entropy       = (4.0, 8.0),
        nesting_variance    = (0.0, 1.0),
    )
    return _make(samples, 'massive_field_selection', 'high')


# ── 5. Huge Argument Lists ───────────────────────────────────────────────────
def huge_argument_lists(n=150, seed=5):
    """
    Attackers pass enormous argument payloads (e.g., huge filter arrays)
    to force server-side processing. Manifests as very large payload_size
    with normal structural features.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 4),
        total_fields        = (1, 10),
        unique_fields       = (1, 8),
        alias_count         = (0, 1),
        introspection_count = 0,
        fragment_count      = 0,
        estimated_cost      = (10, 50),
        payload_size        = (5000, 50000),   # enormous payload despite simple query
        field_entropy       = (0.0, 1.0),
        nesting_variance    = (0.0, 0.5),
    )
    return _make(samples, 'huge_argument_lists', 'high')


# ── 6. Batch Query Abuse ─────────────────────────────────────────────────────
def batch_query_abuse(n=200, seed=6):
    """
    Sends many operations in one request body (Apollo batching).
    Manifests as very high fragment_count + high estimated_cost.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (2, 5),
        total_fields        = (30, 150),
        unique_fields       = (5, 20),
        alias_count         = (0, 5),
        introspection_count = 0,
        fragment_count      = (10, 50),   # many batched operations
        estimated_cost      = (300, 1500),
        payload_size        = (2000, 10000),
        field_entropy       = (1.0, 3.0),
        nesting_variance    = (0.5, 2.0),
    )
    return _make(samples, 'batch_query_abuse', 'high')


# ── 7. Alias Explosion ───────────────────────────────────────────────────────
def alias_explosion(n=200, seed=7):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 4),
        total_fields        = (20, 100),
        unique_fields       = (1, 5),
        alias_count         = (15, 100),
        introspection_count = 0,
        fragment_count      = (0, 3),
        estimated_cost      = (200, 1000),
        payload_size        = (500, 3000),
        field_entropy       = (0.1, 1.0),
        nesting_variance    = (0.0, 0.5),
    )
    return _make(samples, 'alias_explosion', 'high')


# ── 8. IDOR / Field Scanning ─────────────────────────────────────────────────
def idor_field_scanning(n=150, seed=8):
    """
    Attacker iterates over many IDs or enumerates all fields looking for
    unauthorized data. High unique_fields + high field_entropy + sweep pattern.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (2, 5),
        total_fields        = (20, 80),
        unique_fields       = (18, 75),       # very high unique ratio
        alias_count         = (5, 20),
        introspection_count = 0,
        fragment_count      = (0, 2),
        estimated_cost      = (100, 400),
        payload_size        = (400, 2000),
        field_entropy       = (4.5, 8.0),     # max diversity = scanning
        nesting_variance    = (0.5, 2.0),
    )
    return _make(samples, 'idor_field_scanning', 'high')


# ── 9. Broken Auth / Batching Brute Force ────────────────────────────────────
def broken_auth_brute(n=150, seed=9):
    """
    GraphQL-specific brute force: send hundreds of login/mutation attempts
    in one batch. Low field count per op but many operations (fragment_count).
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 3),
        total_fields        = (2, 6),
        unique_fields       = (1, 3),
        alias_count         = (0, 2),
        introspection_count = 0,
        fragment_count      = (50, 500),      # hundreds of batched ops
        estimated_cost      = (100, 1000),
        payload_size        = (3000, 20000),
        field_entropy       = (0.0, 0.8),
        nesting_variance    = (0.0, 0.3),
    )
    return _make(samples, 'broken_auth_brute', 'critical')


# ── 10. SQL / NoSQL Injection ────────────────────────────────────────────────
def sql_nosql_injection(n=150, seed=10):
    """
    Injection payloads in arguments inflate payload_size with unusual
    character distributions. Normal structure with anomalous size/entropy.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 4),
        total_fields        = (1, 8),
        unique_fields       = (1, 6),
        alias_count         = (0, 1),
        introspection_count = 0,
        fragment_count      = 0,
        estimated_cost      = (5, 30),
        payload_size        = (500, 8000),    # injection payload enlarges body
        field_entropy       = (3.0, 7.0),    # weird chars inflate entropy
        nesting_variance    = (0.0, 0.5),
    )
    return _make(samples, 'sql_nosql_injection', 'critical')


# ── 11. OS Command Injection ─────────────────────────────────────────────────
def os_command_injection(n=100, seed=11):
    """
    Command injection via argument strings.
    Similar to SQL injection in feature space but with higher payload_size.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 3),
        total_fields        = (1, 5),
        unique_fields       = (1, 4),
        alias_count         = 0,
        introspection_count = 0,
        fragment_count      = 0,
        estimated_cost      = (5, 20),
        payload_size        = (1000, 15000),  # long injection strings
        field_entropy       = (5.0, 9.0),    # very high char entropy
        nesting_variance    = (0.0, 0.2),
    )
    return _make(samples, 'os_command_injection', 'critical')


# ── 12. SSRF / CRLF Injection ────────────────────────────────────────────────
def ssrf_crlf_injection(n=100, seed=12):
    """
    SSRF via URL arguments / CRLF via header-injecting strings.
    Huge payload, normal structure, extreme field_entropy.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 3),
        total_fields        = (1, 4),
        unique_fields       = (1, 3),
        alias_count         = 0,
        introspection_count = 0,
        fragment_count      = 0,
        estimated_cost      = (3, 15),
        payload_size        = (2000, 20000),
        field_entropy       = (6.0, 10.0),
        nesting_variance    = (0.0, 0.1),
    )
    return _make(samples, 'ssrf_crlf_injection', 'critical')


# ── 13. Introspection Abuse ──────────────────────────────────────────────────
def introspection_abuse(n=200, seed=13):
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (3, 8),
        total_fields        = (10, 50),
        unique_fields       = (8, 45),
        alias_count         = (0, 3),
        introspection_count = (5, 30),
        fragment_count      = (0, 2),
        estimated_cost      = (50, 200),
        payload_size        = (100, 600),
        field_entropy       = (2.0, 4.0),
        nesting_variance    = (1.0, 4.0),
    )
    return _make(samples, 'introspection_abuse', 'medium')


# ── 14. Mutation Spam ────────────────────────────────────────────────────────
def mutation_spam(n=150, seed=14):
    """
    Rapid mutation batching. High total_fields (many write ops),
    high payload_size, moderate depth.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (2, 5),
        total_fields        = (20, 100),
        unique_fields       = (3, 15),
        alias_count         = (10, 50),
        introspection_count = 0,
        fragment_count      = (3, 20),
        estimated_cost      = (400, 1500),
        payload_size        = (1500, 8000),
        field_entropy       = (0.5, 2.0),
        nesting_variance    = (0.5, 2.5),
    )
    return _make(samples, 'mutation_spam', 'high')


# ── 15. Insecure Default Config Abuse ────────────────────────────────────────
def insecure_default_config(n=100, seed=15):
    """
    Exploiting permissive defaults: unlimited query depth, introspection on,
    no rate limiting. Manifests as mixed features near the boundary.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (7, 10),        # just over typical default
        total_fields        = (15, 40),
        unique_fields       = (12, 38),
        alias_count         = (8, 12),
        introspection_count = (1, 5),
        fragment_count      = (1, 4),
        estimated_cost      = (80, 300),
        payload_size        = (200, 800),
        field_entropy       = (2.0, 4.5),
        nesting_variance    = (1.5, 4.0),
    )
    return _make(samples, 'insecure_default_config', 'medium')


# ── 16. Query Frequency Spike ────────────────────────────────────────────────
def query_frequency_spike(n=100, seed=16):
    """
    Simulates the feature profile of flood attacks:
    individually-normal queries but at extreme rate.
    We model this as slightly elevated cost/field combinations
    that represent the 'fingerprint' of automated tooling.
    """
    rng = np.random.default_rng(seed)
    samples = _sample(rng, n,
        max_depth           = (1, 3),
        total_fields        = (1, 5),
        unique_fields       = (1, 4),
        alias_count         = (0, 1),
        introspection_count = 0,
        fragment_count      = 0,
        estimated_cost      = (2, 15),
        payload_size        = (20, 80),
        field_entropy       = (0.0, 0.8),
        nesting_variance    = (0.0, 0.2),
    )
    # Mark as anomalous — frequency is tracked live by EWMA, but structural model
    # also needs to know this pattern is part of flood attacks
    return _make(samples, 'query_frequency_spike', 'medium')


# ── All attacks combined ──────────────────────────────────────────────────────

ALL_ATTACK_GENERATORS = [
    deep_nesting,
    recursive_fragments,
    resolver_cost_explosion,
    massive_field_selection,
    huge_argument_lists,
    batch_query_abuse,
    alias_explosion,
    idor_field_scanning,
    broken_auth_brute,
    sql_nosql_injection,
    os_command_injection,
    ssrf_crlf_injection,
    introspection_abuse,
    mutation_spam,
    insecure_default_config,
    query_frequency_spike,
]


def generate_all_attacks():
    """Return all attack samples as a flat list of dicts."""
    all_samples = []
    for gen in ALL_ATTACK_GENERATORS:
        all_samples.extend(gen())
    return all_samples


if __name__ == '__main__':
    attacks = generate_all_attacks()
    by_type = {}
    for a in attacks:
        by_type.setdefault(a['attack_type'], 0)
        by_type[a['attack_type']] += 1
    print(f"Total attack samples: {len(attacks)}")
    print("\nBy attack type:")
    for t, c in by_type.items():
        print(f"  {t:<30} {c}")
