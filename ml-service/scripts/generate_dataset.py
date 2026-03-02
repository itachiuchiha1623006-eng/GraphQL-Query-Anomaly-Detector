"""
generate_dataset.py
====================
CLI script: generates the full labeled dataset (normal + all 16 attack types)
and exports it as:
  - dataset.csv       (human-readable, for analysis / external ML tools)
  - dataset.json      (for importing into other systems)

Usage:
  python scripts/generate_dataset.py                   # saves to ml-service/data/
  python scripts/generate_dataset.py --output /tmp/ds  # custom output dir
  python scripts/generate_dataset.py --summary          # print stats only
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from ml.training_data import generate_normal_samples, FEATURE_COLUMNS
from ml.attack_generator import generate_all_attacks, ALL_ATTACK_GENERATORS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def build_dataset():
    """Combine normal + all attack samples into one DataFrame."""

    # ── Normal samples ───────────────────────────────────────────────────────
    normal_df = generate_normal_samples(n=3000)
    normal_df['attack_type'] = 'normal'
    normal_df['severity']    = 'none'

    # ── Attack samples ───────────────────────────────────────────────────────
    attack_rows = generate_all_attacks()
    attack_df   = pd.DataFrame(attack_rows)

    # ── Combine ──────────────────────────────────────────────────────────────
    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Ensure all numeric columns are float
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)

    return df


def print_summary(df):
    print("\n" + "="*60)
    print("  GRAPHQL ANOMALY DETECTION — DATASET SUMMARY")
    print("="*60)
    print(f"\n  Total samples      : {len(df):,}")
    print(f"  Normal samples     : {(df['label']==0).sum():,}")
    print(f"  Anomalous samples  : {(df['label']==1).sum():,}")
    print(f"  Attack categories  : {df[df['label']==1]['attack_type'].nunique()}")
    print(f"  Features           : {len(FEATURE_COLUMNS)}")

    print("\n  Breakdown by attack type:")
    print(f"  {'Attack Type':<32} {'Count':>6}  {'Severity'}")
    print("  " + "-"*55)
    for attack_type, grp in df[df['label']==1].groupby('attack_type'):
        severity = grp['severity'].iloc[0]
        print(f"  {attack_type:<32} {len(grp):>6}  {severity}")

    print("\n  Feature ranges (anomalous vs normal):")
    print(f"  {'Feature':<25} {'Normal mean':>13} {'Attack mean':>13}")
    print("  " + "-"*55)
    normal  = df[df['label'] == 0]
    attacks = df[df['label'] == 1]
    for col in FEATURE_COLUMNS:
        print(f"  {col:<25} {normal[col].mean():>13.2f} {attacks[col].mean():>13.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Generate GraphQL anomaly detection dataset')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--summary', action='store_true', help='Print stats only, no file output')
    args = parser.parse_args()

    print("[dataset] Building dataset…")
    df = build_dataset()
    print_summary(df)

    if args.summary:
        return

    os.makedirs(args.output, exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"[dataset] Saved CSV  → {csv_path}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output, 'dataset.json')
    records = df.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump({'meta': {
            'total': len(df),
            'normal': int((df['label']==0).sum()),
            'anomalous': int((df['label']==1).sum()),
            'features': FEATURE_COLUMNS,
            'attack_types': sorted(df[df['label']==1]['attack_type'].unique().tolist()),
        }, 'samples': records}, f, indent=2)
    print(f"[dataset] Saved JSON → {json_path}")

    # ── Per-attack CSV ───────────────────────────────────────────────────────
    per_attack_dir = os.path.join(args.output, 'by_attack_type')
    os.makedirs(per_attack_dir, exist_ok=True)
    for attack_type, grp in df.groupby('attack_type'):
        path = os.path.join(per_attack_dir, f'{attack_type}.csv')
        grp.to_csv(path, index=False)
    print(f"[dataset] Per-type CSVs → {per_attack_dir}/")


if __name__ == '__main__':
    main()
