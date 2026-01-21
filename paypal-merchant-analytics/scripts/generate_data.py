#!/usr/bin/env python3
"""
Data Generation Script

Generates synthetic PayPal-like data for development and testing.
Usage: python scripts/generate_data.py --merchants 5000 --transactions 200000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))

from generator import PayPalDataGenerator, DataConfig


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic PayPal data')
    parser.add_argument('--merchants', type=int, default=3000, help='Number of merchants')
    parser.add_argument('--users', type=int, default=15000, help='Number of users')
    parser.add_argument('--transactions', type=int, default=100000, help='Number of transactions')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PayPal Data Generation Script")
    print("=" * 60)
    
    config = DataConfig(
        num_merchants=args.merchants,
        num_users=args.users,
        num_transactions=args.transactions,
        seed=args.seed
    )
    
    print(f"\nConfiguration:")
    print(f"  Merchants: {config.num_merchants:,}")
    print(f"  Users: {config.num_users:,}")
    print(f"  Transactions: {config.num_transactions:,}")
    print(f"  Output: {args.output}")
    
    generator = PayPalDataGenerator(config)
    generator.save_to_csv(args.output)
    
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
