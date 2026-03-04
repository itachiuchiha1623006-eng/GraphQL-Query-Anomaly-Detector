"""
query_feature_extractor.py
===========================
Converts a raw GraphQL query string into the 10-dimensional feature vector
that the anomaly detection model uses.

Feature columns (matches FEATURE_COLUMNS in training_data.py):
  max_depth, total_fields, unique_fields, alias_count,
  introspection_count, fragment_count, estimated_cost,
  payload_size, field_entropy, nesting_variance

Strategy:
  1. Try graphql-core AST parsing (most accurate).
  2. Fall back to regex heuristics if graphql-core not available.

Usage:
  from ml.query_feature_extractor import extract_features
  features = extract_features("query { user { id name } }")
  # → {"max_depth": 2, "total_fields": 3, ...}
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Optional

# ── Try to import graphql-core ──────────────────────────────────────────────
try:
    from graphql import parse as gql_parse
    from graphql.language.ast import (
        FieldNode, SelectionSetNode, FragmentSpreadNode,
        InlineFragmentNode, OperationDefinitionNode,
        FragmentDefinitionNode,
    )
    _GQL_CORE_AVAILABLE = True
except ImportError:
    _GQL_CORE_AVAILABLE = False


# ── Shared feature column order ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "max_depth",
    "total_fields",
    "unique_fields",
    "alias_count",
    "introspection_count",
    "fragment_count",
    "estimated_cost",
    "payload_size",
    "field_entropy",
    "nesting_variance",
]

_INTROSPECTION_NAMES = {"__schema", "__type", "__typename", "__fields", "__inputFields"}


# ══════════════════════════════════════════════════════════════════════════════
# AST-BASED EXTRACTION (graphql-core)
# ══════════════════════════════════════════════════════════════════════════════

def _walk_selection_set(
    selection_set: Optional["SelectionSetNode"],
    current_depth: int,
    state: dict,
) -> None:
    """Recursively walk the AST selection set, gathering statistics."""
    if selection_set is None:
        return

    state["depth_log"].append(current_depth)
    state["max_depth"] = max(state["max_depth"], current_depth)

    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            state["total_fields"] += 1
            name = selection.name.value
            state["field_names"].append(name)

            if selection.alias is not None:
                state["alias_count"] += 1

            if name in _INTROSPECTION_NAMES or name.startswith("__"):
                state["introspection_count"] += 1

            # Recurse into nested selections
            if selection.selection_set:
                _walk_selection_set(
                    selection.selection_set,
                    current_depth + 1,
                    state,
                )

        elif isinstance(selection, (FragmentSpreadNode, InlineFragmentNode)):
            state["fragment_count"] += 1
            if isinstance(selection, InlineFragmentNode) and selection.selection_set:
                _walk_selection_set(
                    selection.selection_set,
                    current_depth + 1,
                    state,
                )


def _extract_ast(query: str) -> dict:
    """Extract features using graphql-core AST parser."""
    try:
        document = gql_parse(query, allow_legacy_fragment_variables=True)
    except Exception:
        # Parse failure — treat as a suspicious/unusual query
        return _extract_regex(query)

    state = {
        "max_depth":           0,
        "total_fields":        0,
        "alias_count":         0,
        "introspection_count": 0,
        "fragment_count":      0,
        "field_names":         [],
        "depth_log":           [],
    }

    for defn in document.definitions:
        if isinstance(defn, OperationDefinitionNode):
            _walk_selection_set(defn.selection_set, 1, state)
        elif isinstance(defn, FragmentDefinitionNode):
            state["fragment_count"] += 1
            _walk_selection_set(defn.selection_set, 1, state)

    return _finalize(state, len(query))


# ══════════════════════════════════════════════════════════════════════════════
# REGEX FALLBACK EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_regex(query: str) -> dict:
    """
    Heuristic feature extraction using regex — no AST, no dependency.
    Less precise but always works.
    """
    # Payload size
    payload_size = len(query.encode("utf-8"))

    # Strip string literals and comments to avoid false positives
    clean = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', query)  # remove string values
    clean = re.sub(r'#[^\n]*', '', clean)                      # remove comments

    # Depth: count brace opens at each position
    depth = 0
    max_depth = 0
    depth_vals = []
    for ch in clean:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
            depth_vals.append(depth)
        elif ch == '}':
            depth = max(0, depth - 1)

    # Fields: word tokens that look like field names (not keywords, not args)
    # Simple heuristic: identifiers not immediately followed by `(`
    field_candidates = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*[\(:])', clean)
    gql_keywords = {
        "query", "mutation", "subscription", "fragment", "on",
        "true", "false", "null", "type", "interface", "union",
        "enum", "input", "extend", "schema", "directive",
        "implements", "repeatable", "QUERY", "MUTATION",
    }
    field_names = [f for f in field_candidates if f not in gql_keywords]
    total_fields = len(field_names)

    # Aliases: `word: word` pattern
    alias_count = len(re.findall(r'\b[a-zA-Z_]\w*\s*:', clean))

    # Introspection
    introspection_count = len(re.findall(r'__\w+', clean))

    # Fragments
    fragment_count = len(re.findall(r'\.\.\.\s*[a-zA-Z_]', clean))
    fragment_count += len(re.findall(r'\bfragment\b', clean))

    state = {
        "max_depth":           max_depth,
        "total_fields":        total_fields,
        "alias_count":         alias_count,
        "introspection_count": introspection_count,
        "fragment_count":      fragment_count,
        "field_names":         field_names,
        "depth_log":           depth_vals,
    }
    return _finalize(state, payload_size)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FINALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(names: list[str]) -> float:
    """Shannon entropy of field name frequency distribution."""
    if not names:
        return 0.0
    counts = Counter(names)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _nesting_variance(depth_log: list[int]) -> float:
    """Variance of the nesting depth values observed."""
    if len(depth_log) < 2:
        return 0.0
    n = len(depth_log)
    mean = sum(depth_log) / n
    variance = sum((d - mean) ** 2 for d in depth_log) / n
    return round(variance, 4)


def _estimated_cost(total_fields: int, max_depth: int) -> float:
    """Simple complexity heuristic: fields * (1 + depth weighting)."""
    return round(total_fields * (1 + max_depth * 0.5), 2)


def _finalize(state: dict, payload_size: int) -> dict:
    """Turn raw state into the final feature dict."""
    names = state["field_names"]
    depths = state["depth_log"]
    unique_fields = len(set(names))
    total_fields  = state["total_fields"]
    max_depth     = state["max_depth"]

    return {
        "max_depth":           max_depth,
        "total_fields":        total_fields,
        "unique_fields":       unique_fields,
        "alias_count":         state["alias_count"],
        "introspection_count": state["introspection_count"],
        "fragment_count":      state["fragment_count"],
        "estimated_cost":      _estimated_cost(total_fields, max_depth),
        "payload_size":        payload_size,
        "field_entropy":       _shannon_entropy(names),
        "nesting_variance":    _nesting_variance(depths),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(query: str) -> dict:
    """
    Extract the 10-dimensional feature vector from a GraphQL query string.

    Returns a dict with keys matching FEATURE_COLUMNS.
    Always succeeds — falls back to regex if AST parsing fails.

    Args:
        query: Raw GraphQL query string.

    Returns:
        Dict[str, float] with all 10 feature keys.
    """
    if not query or not query.strip():
        return {col: 0.0 for col in FEATURE_COLUMNS}

    if _GQL_CORE_AVAILABLE:
        return _extract_ast(query)
    return _extract_regex(query)


def extract_features_batch(queries: list[str]) -> list[dict]:
    """Extract features from a list of query strings. Returns list of dicts."""
    return [extract_features(q) for q in queries]


# ── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    test_queries = [
        # Normal – simple
        "query { user { id name email } }",

        # Normal – nested
        """query OrderDetails($id: ID!) {
  order(id: $id) {
    id
    status
    total
    items {
      product { name price }
      quantity
    }
  }
}""",

        # Attack – deep nesting
        """query {
  a { b { c { d { e { f { g { h { id } } } } } } } }
}""",

        # Attack – alias abuse
        """query {
  a1: user { id } a2: user { id } a3: user { id }
  a4: user { id } a5: user { id } a6: user { id }
  a7: user { id } a8: user { id } a9: user { id }
}""",

        # Attack – introspection
        "query { __schema { types { name fields { name } } } }",
    ]

    using = "graphql-core AST" if _GQL_CORE_AVAILABLE else "regex fallback"
    print(f"\n[feature_extractor] Using: {using}\n")
    print(f"{'Query':<50}  {'depth':>5} {'fields':>6} {'cost':>6} {'entropy':>7} {'nest_var':>8}")
    print("-" * 85)
    for q in test_queries:
        label = q.split("\n")[0].strip()[:48]
        f = extract_features(q)
        print(
            f"{label:<50}  "
            f"{f['max_depth']:>5.0f} "
            f"{f['total_fields']:>6.0f} "
            f"{f['estimated_cost']:>6.1f} "
            f"{f['field_entropy']:>7.4f} "
            f"{f['nesting_variance']:>8.4f}"
        )
