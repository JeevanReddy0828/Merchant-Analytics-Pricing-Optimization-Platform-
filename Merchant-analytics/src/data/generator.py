"""
PayPal-like Synthetic Data Generator

Generates realistic merchant, transaction, and user behavior data
for analytics and ML model development.

Aligns with JD: "Utilize data mining technologies and various data sources,
such as PayPal user behavior data, inventory data, and transaction data"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import hashlib


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    num_merchants: int = 10000
    num_users: int = 50000
    num_transactions: int = 500000
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    seed: int = 42


class PayPalDataGenerator:
    """
    Generates synthetic PayPal-like data for analytics.
    
    Data includes:
    - Merchant profiles with business categories
    - User behavior patterns
    - Transaction records (batch and real-time simulation)
    - Pricing tier information
    - Cross-border transaction flags
    """
    
    BUSINESS_CATEGORIES = [
        'E-commerce', 'Digital Goods', 'Travel', 'Food & Delivery',
        'Software/SaaS', 'Retail', 'Services', 'Entertainment',
        'Healthcare', 'Education', 'Gaming', 'Subscription'
    ]
    
    REGIONS = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'MEA']
    
    COUNTRIES = {
        'North America': ['US', 'CA', 'MX'],
        'Europe': ['UK', 'DE', 'FR', 'ES', 'IT', 'NL'],
        'Asia Pacific': ['JP', 'AU', 'SG', 'IN', 'KR'],
        'Latin America': ['BR', 'AR', 'CL', 'CO'],
        'MEA': ['AE', 'SA', 'ZA', 'NG']
    }
    
    PRICING_TIERS = ['standard', 'preferred', 'enterprise', 'custom']
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize generator with configuration."""
        self.config = config or DataConfig()
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
        self.merchants_df = None
        self.users_df = None
        self.transactions_df = None
        self.pricing_df = None
        
    def _generate_merchant_id(self, idx: int) -> str:
        """Generate unique merchant ID."""
        return f"M{hashlib.md5(str(idx).encode()).hexdigest()[:12].upper()}"
    
    def _generate_user_id(self, idx: int) -> str:
        """Generate unique user ID."""
        return f"U{hashlib.md5(str(idx + 1000000).encode()).hexdigest()[:12].upper()}"
    
    def generate_merchants(self) -> pd.DataFrame:
        """
        Generate merchant profiles with realistic distributions.
        
        Features:
        - Business category
        - Region and country
        - Account age
        - Pricing tier
        - Monthly volume (log-normal distribution)
        - Churn risk score
        """
        merchants = []
        
        for i in range(self.config.num_merchants):
            region = np.random.choice(self.REGIONS, p=[0.35, 0.30, 0.20, 0.10, 0.05])
            country = np.random.choice(self.COUNTRIES[region])
            
            # Account age affects volume and tier
            account_age_months = int(np.random.exponential(24) + 1)
            
            # Volume follows log-normal (few high-volume merchants)
            base_volume = np.random.lognormal(mean=8, sigma=1.5)
            volume_multiplier = 1 + (account_age_months / 100)
            monthly_volume = base_volume * volume_multiplier
            
            # Pricing tier based on volume
            if monthly_volume > 100000:
                tier = 'enterprise'
            elif monthly_volume > 10000:
                tier = 'preferred'
            elif monthly_volume > 1000:
                tier = 'standard'
            else:
                tier = 'standard'
            
            # Churn risk inversely related to volume and account age
            churn_base = np.random.beta(2, 5)
            churn_risk = churn_base * (1 - min(account_age_months / 48, 0.5))
            
            merchants.append({
                'merchant_id': self._generate_merchant_id(i),
                'business_category': np.random.choice(self.BUSINESS_CATEGORIES),
                'region': region,
                'country': country,
                'account_created': (datetime.now() - timedelta(days=account_age_months * 30)).strftime('%Y-%m-%d'),
                'account_age_months': account_age_months,
                'pricing_tier': tier,
                'monthly_volume_usd': round(monthly_volume, 2),
                'avg_transaction_size': round(np.random.lognormal(mean=3.5, sigma=1), 2),
                'cross_border_enabled': np.random.choice([True, False], p=[0.4, 0.6]),
                'churn_risk_score': round(churn_risk, 4),
                'has_premium_support': tier in ['enterprise', 'preferred'],
                'integration_type': np.random.choice(['API', 'SDK', 'Hosted', 'Plugin'], p=[0.3, 0.25, 0.25, 0.2]),
                'dispute_rate': round(np.random.beta(1, 50), 4),
                'refund_rate': round(np.random.beta(2, 30), 4)
            })
        
        self.merchants_df = pd.DataFrame(merchants)
        return self.merchants_df
    
    def generate_users(self) -> pd.DataFrame:
        """
        Generate user profiles with behavior patterns.
        
        Features:
        - Account status
        - Preferred payment methods
        - Transaction frequency
        - Geographic distribution
        """
        users = []
        
        for i in range(self.config.num_users):
            region = np.random.choice(self.REGIONS, p=[0.40, 0.25, 0.20, 0.10, 0.05])
            country = np.random.choice(self.COUNTRIES[region])
            
            account_age_days = int(np.random.exponential(365) + 1)
            
            users.append({
                'user_id': self._generate_user_id(i),
                'region': region,
                'country': country,
                'account_created': (datetime.now() - timedelta(days=account_age_days)).strftime('%Y-%m-%d'),
                'account_age_days': account_age_days,
                'account_status': np.random.choice(['active', 'inactive', 'suspended'], p=[0.85, 0.12, 0.03]),
                'preferred_payment': np.random.choice(['balance', 'card', 'bank', 'crypto'], p=[0.25, 0.50, 0.20, 0.05]),
                'avg_monthly_transactions': int(np.random.lognormal(mean=1.5, sigma=1)),
                'lifetime_value_usd': round(np.random.lognormal(mean=5, sigma=2), 2),
                'has_verified_email': np.random.choice([True, False], p=[0.95, 0.05]),
                'has_verified_phone': np.random.choice([True, False], p=[0.80, 0.20])
            })
        
        self.users_df = pd.DataFrame(users)
        return self.users_df
    
    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate transaction records with realistic patterns.
        
        Features:
        - Temporal patterns (seasonality, day-of-week effects)
        - Amount distributions by category
        - Cross-border flags
        - Status and outcome
        - Fee calculations
        """
        if self.merchants_df is None:
            self.generate_merchants()
        if self.users_df is None:
            self.generate_users()
        
        transactions = []
        
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        date_range = (end_date - start_date).days
        
        # Sample merchants weighted by volume
        merchant_weights = self.merchants_df['monthly_volume_usd'].values
        merchant_weights = merchant_weights / merchant_weights.sum()
        
        # Sample users weighted by transaction frequency
        user_weights = self.users_df['avg_monthly_transactions'].values + 1
        user_weights = user_weights / user_weights.sum()
        
        for i in range(self.config.num_transactions):
            # Select merchant and user
            merchant_idx = np.random.choice(len(self.merchants_df), p=merchant_weights)
            merchant = self.merchants_df.iloc[merchant_idx]
            
            user_idx = np.random.choice(len(self.users_df), p=user_weights)
            user = self.users_df.iloc[user_idx]
            
            # Generate timestamp with seasonality
            days_offset = np.random.randint(0, date_range)
            hour = int(np.random.normal(14, 4) % 24)  # Peak around 2 PM
            minute = np.random.randint(0, 60)
            timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
            
            # Holiday boost
            if timestamp.month in [11, 12]:  # Holiday season
                amount_multiplier = 1.3
            else:
                amount_multiplier = 1.0
            
            # Weekend adjustment
            if timestamp.weekday() >= 5:
                amount_multiplier *= 0.85
            
            # Transaction amount based on merchant's avg
            base_amount = merchant['avg_transaction_size']
            amount = round(np.random.lognormal(np.log(base_amount), 0.5) * amount_multiplier, 2)
            amount = max(1.0, min(amount, 10000))  # Cap between $1 and $10,000
            
            # Cross-border determination
            is_cross_border = (
                merchant['cross_border_enabled'] and 
                merchant['country'] != user['country'] and
                np.random.random() < 0.3
            )
            
            # Fee calculation based on tier and cross-border
            base_fee_rate = {
                'standard': 0.029,
                'preferred': 0.025,
                'enterprise': 0.022,
                'custom': 0.020
            }[merchant['pricing_tier']]
            
            if is_cross_border:
                fee_rate = base_fee_rate + 0.015  # International fee
            else:
                fee_rate = base_fee_rate
            
            fixed_fee = 0.30
            total_fee = round(amount * fee_rate + fixed_fee, 2)
            
            # Transaction status
            status_probs = [0.95, 0.02, 0.02, 0.01]  # completed, pending, failed, refunded
            status = np.random.choice(['completed', 'pending', 'failed', 'refunded'], p=status_probs)
            
            transactions.append({
                'transaction_id': f"TXN{i:012d}",
                'merchant_id': merchant['merchant_id'],
                'user_id': user['user_id'],
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'date': timestamp.strftime('%Y-%m-%d'),
                'hour': hour,
                'day_of_week': timestamp.weekday(),
                'month': timestamp.month,
                'year': timestamp.year,
                'amount_usd': amount,
                'currency': 'USD' if not is_cross_border else np.random.choice(['USD', 'EUR', 'GBP', 'JPY']),
                'is_cross_border': is_cross_border,
                'merchant_country': merchant['country'],
                'user_country': user['country'],
                'payment_method': user['preferred_payment'],
                'pricing_tier': merchant['pricing_tier'],
                'fee_rate': round(fee_rate, 4),
                'fixed_fee': fixed_fee,
                'total_fee_usd': total_fee,
                'net_amount_usd': round(amount - total_fee, 2),
                'status': status,
                'business_category': merchant['business_category'],
                'platform': np.random.choice(['web', 'mobile_ios', 'mobile_android', 'api'], p=[0.35, 0.25, 0.25, 0.15])
            })
        
        self.transactions_df = pd.DataFrame(transactions)
        return self.transactions_df
    
    def generate_pricing_experiments(self, num_experiments: int = 20) -> pd.DataFrame:
        """
        Generate A/B pricing experiment data.
        
        Features:
        - Control vs treatment groups
        - Revenue impact metrics
        - Statistical significance indicators
        """
        experiments = []
        
        for i in range(num_experiments):
            control_rate = round(np.random.uniform(0.020, 0.035), 4)
            treatment_rate = control_rate + np.random.uniform(-0.005, 0.005)
            
            control_conversions = int(np.random.uniform(1000, 10000))
            treatment_conversions = int(control_conversions * (1 + np.random.uniform(-0.1, 0.15)))
            
            experiments.append({
                'experiment_id': f"EXP{i:04d}",
                'start_date': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d'),
                'duration_days': np.random.randint(14, 60),
                'merchant_segment': np.random.choice(self.BUSINESS_CATEGORIES),
                'control_fee_rate': control_rate,
                'treatment_fee_rate': round(treatment_rate, 4),
                'control_sample_size': control_conversions,
                'treatment_sample_size': treatment_conversions,
                'control_revenue': round(control_conversions * np.random.uniform(50, 200), 2),
                'treatment_revenue': round(treatment_conversions * np.random.uniform(50, 200), 2),
                'p_value': round(np.random.beta(1, 10), 4),
                'is_significant': np.random.choice([True, False], p=[0.3, 0.7]),
                'recommendation': np.random.choice(['adopt', 'reject', 'extend'], p=[0.25, 0.50, 0.25])
            })
        
        self.pricing_df = pd.DataFrame(experiments)
        return self.pricing_df
    
    def generate_merchant_feedback(self, num_feedbacks: int = 5000) -> pd.DataFrame:
        """
        Generate merchant feedback data for NLP analysis.
        
        Features:
        - Support ticket text
        - Sentiment labels
        - Category classification
        """
        feedback_templates = {
            'positive': [
                "Great experience with the checkout process. Our conversion rate improved significantly.",
                "The API integration was smooth and the documentation was very helpful.",
                "Customer support resolved our issue quickly. Very satisfied with the service.",
                "The new dashboard features are excellent for tracking our business metrics.",
                "PayPal has been instrumental in helping us expand internationally."
            ],
            'negative': [
                "Transaction fees are higher than competitors. Need better rates for our volume.",
                "Funds were held for too long without clear explanation.",
                "The dispute resolution process took too long and we lost money.",
                "Integration issues caused downtime for our checkout.",
                "Customer support response time needs improvement."
            ],
            'neutral': [
                "Looking for information about enterprise pricing options.",
                "Need clarification on the cross-border fee structure.",
                "Requesting documentation for webhook implementation.",
                "Inquiry about new features on the roadmap.",
                "General question about account verification process."
            ]
        }
        
        categories = ['billing', 'technical', 'account', 'disputes', 'features', 'general']
        
        feedbacks = []
        for i in range(num_feedbacks):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.25, 0.35])
            text = np.random.choice(feedback_templates[sentiment])
            
            # Add some noise/variation
            if np.random.random() < 0.3:
                text = text + f" Reference: TKT{np.random.randint(10000, 99999)}"
            
            feedbacks.append({
                'feedback_id': f"FB{i:06d}",
                'merchant_id': self._generate_merchant_id(np.random.randint(0, self.config.num_merchants)),
                'timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
                'text': text,
                'sentiment_label': sentiment,
                'category': np.random.choice(categories),
                'priority': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.35, 0.15]),
                'resolved': np.random.choice([True, False], p=[0.75, 0.25])
            })
        
        return pd.DataFrame(feedbacks)
    
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets and return as dictionary."""
        return {
            'merchants': self.generate_merchants(),
            'users': self.generate_users(),
            'transactions': self.generate_transactions(),
            'pricing_experiments': self.generate_pricing_experiments(),
            'feedback': self.generate_merchant_feedback()
        }
    
    def save_to_csv(self, output_dir: str = 'data/raw') -> None:
        """Save all generated data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        data = self.generate_all()
        for name, df in data.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {name}: {len(df)} records to {filepath}")


def main():
    """Generate sample data for development."""
    config = DataConfig(
        num_merchants=5000,
        num_users=25000,
        num_transactions=250000
    )
    
    generator = PayPalDataGenerator(config)
    generator.save_to_csv('data/raw')
    
    # Print summary statistics
    print("\n=== Data Generation Summary ===")
    print(f"Merchants: {len(generator.merchants_df)}")
    print(f"Users: {len(generator.users_df)}")
    print(f"Transactions: {len(generator.transactions_df)}")
    print(f"\nMerchant distribution by tier:")
    print(generator.merchants_df['pricing_tier'].value_counts())
    print(f"\nTransaction status distribution:")
    print(generator.transactions_df['status'].value_counts(normalize=True))


if __name__ == "__main__":
    main()
