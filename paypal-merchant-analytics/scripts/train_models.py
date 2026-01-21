#!/usr/bin/env python3
"""
Model Training Script

Train and save ML models for pricing and churn prediction.
Usage: python scripts/train_models.py --model pricing --data data/raw
"""

import argparse
import sys
import pickle
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'models'))

import pandas as pd


def train_pricing_model(data_dir: str, output_dir: str):
    """Train pricing optimization model."""
    from pricing_optimizer import PricingOptimizer
    
    print("\n--- Training Pricing Model ---")
    
    # Load data
    merchants = pd.read_csv(f'{data_dir}/merchants.csv')
    transactions = pd.read_csv(f'{data_dir}/transactions.csv')
    
    print(f"Loaded {len(merchants):,} merchants, {len(transactions):,} transactions")
    
    # Train model
    model = PricingOptimizer()
    metrics = model.fit(merchants, transactions)
    
    print(f"\nModel Performance:")
    print(f"  MAE: {metrics.mae:.4f}")
    print(f"  R²: {metrics.r2:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f'pricing_model_{datetime.now().strftime("%Y%m%d")}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_file}")
    return model, metrics


def train_churn_model(data_dir: str, output_dir: str):
    """Train churn prediction model."""
    from churn_predictor import ChurnPredictor
    
    print("\n--- Training Churn Model ---")
    
    # Load data
    merchants = pd.read_csv(f'{data_dir}/merchants.csv')
    transactions = pd.read_csv(f'{data_dir}/transactions.csv')
    
    print(f"Loaded {len(merchants):,} merchants, {len(transactions):,} transactions")
    
    # Train model
    model = ChurnPredictor()
    metrics = model.fit(merchants, transactions)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
    print(f"  F1: {metrics.f1:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f'churn_model_{datetime.now().strftime("%Y%m%d")}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_file}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--model', type=str, choices=['pricing', 'churn', 'all'], 
                        default='all', help='Model to train')
    parser.add_argument('--data', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--output', type=str, default='models/trained', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PayPal Model Training Script")
    print("=" * 60)
    
    if args.model in ['pricing', 'all']:
        train_pricing_model(args.data, args.output)
    
    if args.model in ['churn', 'all']:
        train_churn_model(args.data, args.output)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
