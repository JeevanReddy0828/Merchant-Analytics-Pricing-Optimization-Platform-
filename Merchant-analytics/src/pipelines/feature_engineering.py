"""
Feature Engineering Module

Centralized feature transformations for ML models.
Ensures consistency between training and inference.

Aligns with JD: "Perform exploratory data analysis to understand 
the correlation between target performance and platform features"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    scale_numerical: bool = True
    encode_categorical: bool = True


class FeatureTransformer:
    """
    Feature transformation pipeline.
    
    Handles:
    - Numerical scaling
    - Categorical encoding
    - Missing value imputation
    - Feature creation
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_stats: Dict[str, dict] = {}
        
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        """Fit transformers on training data."""
        logger.info("Fitting feature transformer...")
        
        # Fit numerical scaler
        if self.config.scale_numerical and self.config.numerical_features:
            available_num = [c for c in self.config.numerical_features if c in df.columns]
            if available_num:
                self.scaler.fit(df[available_num].fillna(0))
                
                # Store statistics
                for i, col in enumerate(available_num):
                    self.feature_stats[col] = {
                        'mean': self.scaler.mean_[i],
                        'std': self.scaler.scale_[i]
                    }
        
        # Fit categorical encoders
        if self.config.encode_categorical and self.config.categorical_features:
            for col in self.config.categorical_features:
                if col in df.columns:
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(df[col].astype(str).fillna('unknown'))
                    
                    self.feature_stats[col] = {
                        'classes': list(self.encoders[col].classes_),
                        'n_classes': len(self.encoders[col].classes_)
                    }
        
        self.is_fitted = True
        logger.info(f"Transformer fitted with {len(self.feature_stats)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        result = df.copy()
        
        # Scale numerical features
        if self.config.scale_numerical and self.config.numerical_features:
            available_num = [c for c in self.config.numerical_features if c in result.columns]
            if available_num:
                scaled = self.scaler.transform(result[available_num].fillna(0))
                for i, col in enumerate(available_num):
                    result[f'{col}_scaled'] = scaled[:, i]
        
        # Encode categorical features
        if self.config.encode_categorical:
            for col, encoder in self.encoders.items():
                if col in result.columns:
                    # Handle unseen categories
                    result[f'{col}_encoded'] = result[col].apply(
                        lambda x: encoder.transform([str(x)])[0] 
                        if str(x) in encoder.classes_ else -1
                    )
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_feature_names(self, include_original: bool = False) -> List[str]:
        """Get list of transformed feature names."""
        features = []
        
        if self.config.scale_numerical:
            features.extend([f'{c}_scaled' for c in self.config.numerical_features])
        
        if self.config.encode_categorical:
            features.extend([f'{c}_encoded' for c in self.config.categorical_features])
        
        if include_original:
            features.extend(self.config.numerical_features)
            features.extend(self.config.categorical_features)
        
        return features


class MerchantFeatureBuilder:
    """
    Build features specifically for merchant analytics.
    
    Creates domain-specific features for:
    - Pricing optimization
    - Churn prediction
    - Risk assessment
    """
    
    @staticmethod
    def build_volume_features(
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build volume-related features."""
        logger.info("Building volume features...")
        
        # Aggregate transaction volume
        volume_agg = transactions_df.groupby('merchant_id').agg({
            'amount_usd': ['sum', 'mean', 'std', 'count'],
            'total_fee_usd': 'sum'
        })
        volume_agg.columns = [
            'total_volume', 'avg_transaction', 'txn_std', 
            'transaction_count', 'total_fees'
        ]
        volume_agg = volume_agg.reset_index()
        
        # Merge with merchants
        result = merchants_df.merge(volume_agg, on='merchant_id', how='left')
        result = result.fillna(0)
        
        # Derived features
        result['volume_per_txn'] = result['total_volume'] / result['transaction_count'].replace(0, 1)
        result['fee_rate_actual'] = result['total_fees'] / result['total_volume'].replace(0, 1)
        result['volume_volatility'] = result['txn_std'] / result['avg_transaction'].replace(0, 1)
        
        return result
    
    @staticmethod
    def build_temporal_features(
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build time-based features."""
        logger.info("Building temporal features...")
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        max_date = transactions_df['date'].max()
        min_date = transactions_df['date'].min()
        
        # Recency
        recency = transactions_df.groupby('merchant_id')['date'].max().reset_index()
        recency.columns = ['merchant_id', 'last_txn_date']
        recency['days_since_last_txn'] = (max_date - recency['last_txn_date']).dt.days
        
        # Frequency
        freq = transactions_df.groupby('merchant_id').agg({
            'date': ['min', 'count']
        })
        freq.columns = ['first_txn_date', 'total_txns']
        freq = freq.reset_index()
        freq['active_days'] = (max_date - freq['first_txn_date']).dt.days + 1
        freq['txn_frequency'] = freq['total_txns'] / freq['active_days'].replace(0, 1)
        
        # Merge
        result = merchants_df.merge(
            recency[['merchant_id', 'days_since_last_txn']], 
            on='merchant_id', how='left'
        )
        result = result.merge(
            freq[['merchant_id', 'active_days', 'txn_frequency']], 
            on='merchant_id', how='left'
        )
        result = result.fillna(0)
        
        return result
    
    @staticmethod
    def build_trend_features(
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """Build trend features comparing recent vs historical."""
        logger.info("Building trend features...")
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        max_date = transactions_df['date'].max()
        
        # Recent period
        recent_start = max_date - pd.Timedelta(days=lookback_days)
        recent = transactions_df[transactions_df['date'] > recent_start]
        
        # Prior period
        prior_end = recent_start
        prior_start = prior_end - pd.Timedelta(days=lookback_days)
        prior = transactions_df[
            (transactions_df['date'] > prior_start) & 
            (transactions_df['date'] <= prior_end)
        ]
        
        # Aggregate both periods
        recent_agg = recent.groupby('merchant_id')['amount_usd'].sum().reset_index()
        recent_agg.columns = ['merchant_id', 'recent_volume']
        
        prior_agg = prior.groupby('merchant_id')['amount_usd'].sum().reset_index()
        prior_agg.columns = ['merchant_id', 'prior_volume']
        
        # Merge and calculate trend
        result = merchants_df.merge(recent_agg, on='merchant_id', how='left')
        result = result.merge(prior_agg, on='merchant_id', how='left')
        result = result.fillna(0)
        
        result['volume_trend'] = (
            result['recent_volume'] / result['prior_volume'].replace(0, 1) - 1
        )
        result['volume_trend'] = result['volume_trend'].clip(-1, 5)  # Cap extreme values
        
        return result
    
    @staticmethod
    def build_cross_border_features(
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build cross-border transaction features."""
        logger.info("Building cross-border features...")
        
        # Cross-border metrics
        xb_agg = transactions_df.groupby('merchant_id').agg({
            'is_cross_border': ['sum', 'mean'],
            'transaction_id': 'count'
        })
        xb_agg.columns = ['xb_count', 'xb_ratio', 'total_txns']
        xb_agg = xb_agg.reset_index()
        
        # Cross-border volume
        xb_volume = transactions_df[transactions_df['is_cross_border'] == True].groupby(
            'merchant_id'
        )['amount_usd'].sum().reset_index()
        xb_volume.columns = ['merchant_id', 'xb_volume']
        
        # Merge
        result = merchants_df.merge(xb_agg, on='merchant_id', how='left')
        result = result.merge(xb_volume, on='merchant_id', how='left')
        result = result.fillna(0)
        
        result['xb_avg_size'] = result['xb_volume'] / result['xb_count'].replace(0, 1)
        
        return result


def create_ml_features(
    merchants_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    target: str = 'pricing'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create complete feature set for ML models.
    
    Args:
        merchants_df: Merchant profiles
        transactions_df: Transaction records
        target: 'pricing' or 'churn'
        
    Returns:
        Tuple of (feature DataFrame, feature column names)
    """
    builder = MerchantFeatureBuilder()
    
    # Build all feature sets
    volume_features = builder.build_volume_features(merchants_df, transactions_df)
    temporal_features = builder.build_temporal_features(merchants_df, transactions_df)
    trend_features = builder.build_trend_features(merchants_df, transactions_df)
    xb_features = builder.build_cross_border_features(merchants_df, transactions_df)
    
    # Merge all features
    result = merchants_df[['merchant_id']].copy()
    
    for feature_df in [volume_features, temporal_features, trend_features, xb_features]:
        merge_cols = [c for c in feature_df.columns if c not in result.columns or c == 'merchant_id']
        result = result.merge(feature_df[merge_cols], on='merchant_id', how='left')
    
    # Add original merchant features
    result = result.merge(merchants_df, on='merchant_id', how='left')
    
    # Define feature columns based on target
    if target == 'pricing':
        feature_cols = [
            'monthly_volume_usd', 'avg_transaction', 'volume_volatility',
            'churn_risk_score', 'dispute_rate', 'refund_rate',
            'xb_ratio', 'days_since_last_txn', 'txn_frequency',
            'volume_trend', 'account_age_months'
        ]
    else:  # churn
        feature_cols = [
            'days_since_last_txn', 'txn_frequency', 'volume_trend',
            'total_volume', 'avg_transaction', 'account_age_months',
            'dispute_rate', 'refund_rate', 'xb_ratio'
        ]
    
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in result.columns]
    
    return result, available_cols


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.insert(0, '../data')
    from generator import PayPalDataGenerator, DataConfig
    
    config = DataConfig(num_merchants=500, num_users=2000, num_transactions=10000)
    generator = PayPalDataGenerator(config)
    data = generator.generate_all()
    
    features_df, feature_cols = create_ml_features(
        data['merchants'], 
        data['transactions'],
        target='pricing'
    )
    
    print(f"Created features: {len(features_df)} records")
    print(f"Feature columns: {feature_cols}")
    print(f"\nFeature statistics:")
    print(features_df[feature_cols].describe())
