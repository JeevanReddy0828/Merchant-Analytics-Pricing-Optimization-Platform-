"""
Batch Data Pipeline

Daily batch processing pipeline for merchant analytics.
Handles data ingestion, transformation, and model updates.

Aligns with JD: "Implement large-scale data mining and machine learning 
algorithms and data model pipelines in a software production environment"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for batch pipeline."""
    input_dir: str = 'data/raw'
    output_dir: str = 'data/processed'
    model_dir: str = 'models/trained'
    report_dir: str = 'reports/daily'
    batch_date: Optional[str] = None  # YYYY-MM-DD, defaults to yesterday
    
    def __post_init__(self):
        if self.batch_date is None:
            self.batch_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


class DataIngestionStep:
    """
    Step 1: Data Ingestion
    
    Loads raw data from various sources and performs
    initial validation and deduplication.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.input_path = Path(config.input_dir)
        
    def execute(self) -> Dict[str, pd.DataFrame]:
        """Execute data ingestion step."""
        logger.info("Starting data ingestion...")
        
        data = {}
        
        # Load each data source
        for file_path in self.input_path.glob('*.csv'):
            table_name = file_path.stem
            logger.info(f"Loading {table_name}...")
            
            df = pd.read_csv(file_path)
            
            # Basic validation
            initial_count = len(df)
            df = df.drop_duplicates()
            dedup_count = len(df)
            
            if initial_count != dedup_count:
                logger.warning(f"Removed {initial_count - dedup_count} duplicates from {table_name}")
            
            data[table_name] = df
            logger.info(f"Loaded {table_name}: {len(df):,} records")
        
        return data


class DataTransformationStep:
    """
    Step 2: Data Transformation
    
    Applies business logic transformations, creates
    derived features, and prepares data for analytics.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def execute(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Execute data transformation step."""
        logger.info("Starting data transformation...")
        
        transformed = {}
        
        # Transform transactions
        if 'transactions' in data:
            transformed['transactions'] = self._transform_transactions(data['transactions'])
        
        # Transform merchants
        if 'merchants' in data:
            transformed['merchants'] = self._transform_merchants(
                data['merchants'],
                transformed.get('transactions', data.get('transactions'))
            )
        
        # Copy other tables
        for key, df in data.items():
            if key not in transformed:
                transformed[key] = df
        
        return transformed
    
    def _transform_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform transaction data."""
        logger.info("Transforming transactions...")
        
        result = df.copy()
        
        # Parse dates
        result['date'] = pd.to_datetime(result['date'])
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        
        # Add time-based features
        result['week'] = result['date'].dt.isocalendar().week
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        result['is_holiday_season'] = result['month'].isin([11, 12]).astype(int)
        
        # Add fee analysis features
        result['effective_rate'] = result['total_fee_usd'] / result['amount_usd']
        result['fee_burden'] = (result['effective_rate'] > 0.03).astype(int)
        
        logger.info(f"Transformed {len(result):,} transactions")
        return result
    
    def _transform_merchants(
        self, 
        merchants_df: pd.DataFrame, 
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform merchant data with transaction aggregates."""
        logger.info("Transforming merchants...")
        
        result = merchants_df.copy()
        
        if transactions_df is not None:
            # Aggregate transaction metrics
            txn_agg = transactions_df.groupby('merchant_id').agg({
                'transaction_id': 'count',
                'amount_usd': ['sum', 'mean', 'std'],
                'total_fee_usd': 'sum',
                'is_cross_border': 'mean',
                'status': lambda x: (x == 'completed').mean()
            })
            
            txn_agg.columns = [
                'total_transactions', 'total_volume', 'avg_txn_size', 'txn_std',
                'total_fees', 'cross_border_ratio', 'success_rate'
            ]
            txn_agg = txn_agg.reset_index()
            
            # Merge
            result = result.merge(txn_agg, on='merchant_id', how='left')
            result = result.fillna(0)
        
        # Add derived features
        result['revenue_per_month'] = result.get('total_fees', 0) / 12
        result['volume_tier'] = pd.cut(
            result['monthly_volume_usd'],
            bins=[0, 1000, 10000, 100000, float('inf')],
            labels=['micro', 'small', 'medium', 'large']
        )
        
        logger.info(f"Transformed {len(result):,} merchants")
        return result


class FeatureEngineeringStep:
    """
    Step 3: Feature Engineering
    
    Creates ML-ready features for model training
    and scoring.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def execute(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Execute feature engineering step."""
        logger.info("Starting feature engineering...")
        
        features = {}
        
        # Pricing features
        if 'merchants' in data and 'transactions' in data:
            features['pricing_features'] = self._create_pricing_features(
                data['merchants'], data['transactions']
            )
            
            features['churn_features'] = self._create_churn_features(
                data['merchants'], data['transactions']
            )
        
        # Combine with original data
        features.update(data)
        
        return features
    
    def _create_pricing_features(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features for pricing model."""
        logger.info("Creating pricing features...")
        
        # Start with merchant base
        features = merchants_df[['merchant_id', 'business_category', 'region', 
                                  'pricing_tier', 'monthly_volume_usd', 
                                  'churn_risk_score']].copy()
        
        # Add transaction-based features
        txn_features = transactions_df.groupby('merchant_id').agg({
            'amount_usd': ['mean', 'std', 'median'],
            'is_cross_border': 'mean',
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 12,
            'platform': lambda x: x.mode().iloc[0] if len(x) > 0 else 'web'
        })
        
        txn_features.columns = [
            'avg_amount', 'amount_std', 'median_amount',
            'cross_border_ratio', 'peak_hour', 'primary_platform'
        ]
        txn_features = txn_features.reset_index()
        
        features = features.merge(txn_features, on='merchant_id', how='left')
        features = features.fillna(0)
        
        logger.info(f"Created pricing features: {len(features):,} records, {len(features.columns)} features")
        return features
    
    def _create_churn_features(
        self,
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features for churn model."""
        logger.info("Creating churn features...")
        
        max_date = pd.to_datetime(transactions_df['date']).max()
        
        # Recency features
        recency = transactions_df.groupby('merchant_id').agg({
            'date': 'max',
            'transaction_id': 'count'
        }).reset_index()
        recency.columns = ['merchant_id', 'last_txn_date', 'total_txns']
        recency['last_txn_date'] = pd.to_datetime(recency['last_txn_date'])
        recency['days_since_last_txn'] = (max_date - recency['last_txn_date']).dt.days
        
        # Frequency features
        txn_dates = transactions_df.groupby('merchant_id')['date'].apply(
            lambda x: pd.to_datetime(x).diff().dt.days.mean()
        ).reset_index()
        txn_dates.columns = ['merchant_id', 'avg_days_between_txn']
        
        # Merge all
        features = merchants_df[['merchant_id', 'account_age_months', 'pricing_tier',
                                  'dispute_rate', 'refund_rate', 'churn_risk_score']].copy()
        features = features.merge(recency, on='merchant_id', how='left')
        features = features.merge(txn_dates, on='merchant_id', how='left')
        features = features.fillna(0)
        
        logger.info(f"Created churn features: {len(features):,} records")
        return features


class AggregationStep:
    """
    Step 4: Aggregation
    
    Creates daily/weekly/monthly aggregates for
    reporting and dashboards.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def execute(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Execute aggregation step."""
        logger.info("Starting aggregation...")
        
        aggregates = {}
        
        if 'transactions' in data:
            transactions = data['transactions']
            
            # Daily aggregates
            aggregates['daily_summary'] = self._create_daily_summary(transactions)
            
            # Category aggregates
            aggregates['category_summary'] = self._create_category_summary(
                transactions, data.get('merchants')
            )
        
        # Save aggregates
        for name, df in aggregates.items():
            output_file = self.output_path / f'{name}_{self.config.batch_date}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {name} to {output_file}")
        
        return {**data, **aggregates}
    
    def _create_daily_summary(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create daily transaction summary."""
        daily = transactions.groupby('date').agg({
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'total_fee_usd': 'sum',
            'merchant_id': 'nunique',
            'user_id': 'nunique',
            'is_cross_border': 'sum'
        })
        
        daily.columns = [
            'transaction_count', 'total_volume', 'avg_amount',
            'total_fees', 'active_merchants', 'active_users',
            'cross_border_count'
        ]
        
        return daily.reset_index()
    
    def _create_category_summary(
        self, 
        transactions: pd.DataFrame,
        merchants: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Create category-level summary."""
        if merchants is None:
            return pd.DataFrame()
        
        merged = transactions.merge(
            merchants[['merchant_id', 'business_category']],
            on='merchant_id',
            how='left'
        )
        
        category = merged.groupby('business_category').agg({
            'transaction_id': 'count',
            'amount_usd': 'sum',
            'total_fee_usd': 'sum',
            'merchant_id': 'nunique'
        })
        
        category.columns = ['transactions', 'volume', 'fees', 'merchants']
        
        return category.reset_index()


class BatchPipeline:
    """
    Main batch pipeline orchestrator.
    
    Coordinates execution of all pipeline steps
    and handles error recovery.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize steps
        self.steps = [
            ('ingestion', DataIngestionStep(self.config)),
            ('transformation', DataTransformationStep(self.config)),
            ('feature_engineering', FeatureEngineeringStep(self.config)),
            ('aggregation', AggregationStep(self.config))
        ]
        
    def run(self) -> Dict[str, pd.DataFrame]:
        """Run the complete batch pipeline."""
        logger.info(f"Starting batch pipeline for {self.config.batch_date}")
        start_time = datetime.now()
        
        data = {}
        
        for step_name, step in self.steps:
            try:
                logger.info(f"Executing step: {step_name}")
                
                if step_name == 'ingestion':
                    data = step.execute()
                else:
                    data = step.execute(data)
                    
                logger.info(f"Completed step: {step_name}")
                
            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                raise
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed in {elapsed:.1f} seconds")
        
        # Save pipeline metadata
        self._save_metadata(data, elapsed)
        
        return data
    
    def _save_metadata(self, data: Dict[str, pd.DataFrame], elapsed: float):
        """Save pipeline execution metadata."""
        metadata = {
            'batch_date': self.config.batch_date,
            'execution_time': datetime.now().isoformat(),
            'duration_seconds': elapsed,
            'tables_processed': {
                name: len(df) for name, df in data.items()
            }
        }
        
        metadata_path = Path(self.config.output_dir) / f'metadata_{self.config.batch_date}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Run batch pipeline."""
    config = PipelineConfig(
        input_dir='data/raw',
        output_dir='data/processed'
    )
    
    pipeline = BatchPipeline(config)
    data = pipeline.run()
    
    print("\n=== Pipeline Summary ===")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} records")


if __name__ == "__main__":
    main()
