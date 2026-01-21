"""
Dynamic Pricing Optimization Model

Machine learning model for optimizing merchant pricing based on:
- Transaction patterns
- Merchant characteristics
- Market conditions
- Competitive positioning

Aligns with JD: "Build data analytics and statistical models focusing on 
pricing optimization initiatives and their impact on merchant behavior"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using sklearn GradientBoosting")


@dataclass
class PricingFeatures:
    """Feature configuration for pricing model."""
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    target: str = 'optimal_fee_rate'
    
    def __post_init__(self):
        if self.categorical_features is None:
            self.categorical_features = [
                'business_category', 'region', 'pricing_tier', 'integration_type'
            ]
        if self.numerical_features is None:
            self.numerical_features = [
                'monthly_volume_usd', 'avg_transaction_size', 'account_age_months',
                'cross_border_ratio', 'churn_risk_score', 'dispute_rate', 
                'refund_rate', 'transaction_count'
            ]


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    cv_scores: np.ndarray
    feature_importance: Dict[str, float]


class PricingOptimizer:
    """
    ML-based dynamic pricing optimization engine.
    
    Capabilities:
    - Optimal fee rate prediction
    - Price elasticity estimation
    - A/B test simulation
    - Revenue impact forecasting
    """
    
    def __init__(self, feature_config: Optional[PricingFeatures] = None):
        """
        Initialize pricing optimizer.
        
        Args:
            feature_config: Feature configuration object
        """
        self.feature_config = feature_config or PricingFeatures()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        fit_encoders: bool = False
    ) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: Input DataFrame
            fit_encoders: Whether to fit label encoders
            
        Returns:
            Prepared feature DataFrame
        """
        result = df.copy()
        
        # Encode categorical features
        for col in self.feature_config.categorical_features:
            if col in result.columns:
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    result[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        result[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        result[f'{col}_encoded'] = result[col].apply(
                            lambda x: self.label_encoders[col].transform([str(x)])[0]
                            if str(x) in self.label_encoders[col].classes_
                            else -1
                        )
        
        return result
    
    def _compute_optimal_rate(self, merchants_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute optimal fee rate based on historical data.
        
        The "optimal" rate maximizes revenue while minimizing churn risk.
        
        Args:
            merchants_df: Merchant profiles
            transactions_df: Transaction history
            
        Returns:
            DataFrame with optimal rate targets
        """
        # Aggregate transaction metrics per merchant
        txn_agg = transactions_df.groupby('merchant_id').agg({
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'total_fee_usd': 'sum',
            'fee_rate': 'mean',
            'is_cross_border': 'mean'
        }).reset_index()
        
        txn_agg.columns = [
            'merchant_id', 'transaction_count', 'total_volume', 
            'avg_transaction_size', 'total_fees', 'current_fee_rate',
            'cross_border_ratio'
        ]
        
        # Merge with merchant data
        merged = merchants_df.merge(txn_agg, on='merchant_id', how='left')
        merged = merged.fillna(0)
        
        # Compute optimal rate based on multiple factors
        # Higher volume -> lower rate (volume discount)
        # Higher churn risk -> lower rate (retention)
        # Higher cross-border -> higher rate (complexity)
        # Lower dispute rate -> lower rate (quality merchant)
        
        base_rate = 0.029  # Standard rate
        
        # Volume adjustment (up to -0.7% for high volume)
        volume_adj = -np.clip(
            np.log1p(merged['monthly_volume_usd']) / 50, 
            0, 0.007
        )
        
        # Churn risk adjustment (up to -0.3% for high risk)
        churn_adj = -merged['churn_risk_score'] * 0.003
        
        # Cross-border adjustment (up to +1% for international)
        xb_adj = merged['cross_border_ratio'] * 0.01
        
        # Quality adjustment (dispute/refund rate penalty)
        quality_adj = (merged['dispute_rate'] + merged['refund_rate']) * 0.005
        
        # Compute optimal rate
        merged['optimal_fee_rate'] = np.clip(
            base_rate + volume_adj + churn_adj + xb_adj + quality_adj,
            0.015,  # Min rate
            0.045   # Max rate
        )
        
        return merged
    
    def fit(
        self, 
        merchants_df: pd.DataFrame, 
        transactions_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> ModelMetrics:
        """
        Train the pricing optimization model.
        
        Args:
            merchants_df: Merchant profiles DataFrame
            transactions_df: Transaction history DataFrame
            test_size: Fraction for test set
            
        Returns:
            ModelMetrics with evaluation results
        """
        # Prepare training data
        training_data = self._compute_optimal_rate(merchants_df, transactions_df)
        training_data = self._prepare_features(training_data, fit_encoders=True)
        
        # Build feature matrix
        feature_cols = (
            self.feature_config.numerical_features + 
            [f'{col}_encoded' for col in self.feature_config.categorical_features 
             if f'{col}_encoded' in training_data.columns]
        )
        
        # Filter to available columns
        available_features = [c for c in feature_cols if c in training_data.columns]
        self.feature_names = available_features
        
        X = training_data[available_features].fillna(0)
        y = training_data[self.feature_config.target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Initialize model
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error'
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
        else:
            importance = {}
        
        metrics = ModelMetrics(
            mae=mean_absolute_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            r2=r2_score(y_test, y_pred),
            cv_scores=-cv_scores,
            feature_importance=importance
        )
        
        return metrics
    
    def predict(self, merchant_data: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal fee rates for merchants.
        
        Args:
            merchant_data: DataFrame with merchant features
            
        Returns:
            Array of predicted optimal fee rates
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        prepared = self._prepare_features(merchant_data, fit_encoders=False)
        
        # Build feature matrix
        available_features = [c for c in self.feature_names if c in prepared.columns]
        X = prepared[available_features].fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range
        return np.clip(predictions, 0.015, 0.045)
    
    def simulate_price_change(
        self, 
        merchant_data: pd.DataFrame,
        current_rates: np.ndarray,
        new_rates: np.ndarray,
        volume: np.ndarray
    ) -> Dict[str, float]:
        """
        Simulate revenue impact of price changes.
        
        Args:
            merchant_data: Merchant features
            current_rates: Current fee rates
            new_rates: Proposed new rates
            volume: Expected transaction volume
            
        Returns:
            Dictionary with simulation results
        """
        # Price elasticity estimation (simplified)
        # Assume -0.5 elasticity for fee rate changes
        elasticity = -0.5
        
        rate_change = (new_rates - current_rates) / current_rates
        volume_impact = 1 + (rate_change * elasticity)
        
        # Revenue calculation
        current_revenue = (current_rates * volume).sum()
        projected_volume = volume * volume_impact
        projected_revenue = (new_rates * projected_volume).sum()
        
        return {
            'current_revenue': round(current_revenue, 2),
            'projected_revenue': round(projected_revenue, 2),
            'revenue_change': round(projected_revenue - current_revenue, 2),
            'revenue_change_pct': round((projected_revenue / current_revenue - 1) * 100, 2),
            'avg_rate_change': round(rate_change.mean() * 100, 2),
            'merchants_affected': len(merchant_data)
        }
    
    def get_pricing_recommendations(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Generate pricing recommendations for merchants.
        
        Args:
            merchants_df: Merchant profiles
            transactions_df: Transaction history
            top_n: Number of top recommendations
            
        Returns:
            DataFrame with pricing recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data
        prepared = self._compute_optimal_rate(merchants_df, transactions_df)
        
        # Get predictions
        optimal_rates = self.predict(prepared)
        
        # Current rates (using tier defaults)
        tier_rates = {
            'standard': 0.029,
            'preferred': 0.025,
            'enterprise': 0.022,
            'custom': 0.020
        }
        prepared['current_rate'] = prepared['pricing_tier'].map(tier_rates).fillna(0.029)
        prepared['recommended_rate'] = optimal_rates
        prepared['rate_change'] = prepared['recommended_rate'] - prepared['current_rate']
        
        # Calculate revenue impact
        prepared['monthly_fee_current'] = prepared['monthly_volume_usd'] * prepared['current_rate']
        prepared['monthly_fee_recommended'] = prepared['monthly_volume_usd'] * prepared['recommended_rate']
        prepared['revenue_impact'] = prepared['monthly_fee_recommended'] - prepared['monthly_fee_current']
        
        # Sort by absolute revenue impact
        recommendations = prepared.nlargest(top_n, 'revenue_impact', keep='first')
        
        return recommendations[[
            'merchant_id', 'business_category', 'pricing_tier',
            'monthly_volume_usd', 'churn_risk_score',
            'current_rate', 'recommended_rate', 'rate_change',
            'monthly_fee_current', 'monthly_fee_recommended', 'revenue_impact'
        ]].round(4)


class PriceElasticityAnalyzer:
    """
    Analyze price sensitivity across merchant segments.
    
    Uses historical data to estimate elasticity coefficients
    for different merchant segments.
    """
    
    def __init__(self):
        """Initialize elasticity analyzer."""
        self.elasticity_estimates: Dict[str, float] = {}
    
    def estimate_elasticity(
        self, 
        pricing_experiments: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Estimate price elasticity from A/B test data.
        
        Args:
            pricing_experiments: DataFrame with experiment results
            
        Returns:
            Dictionary of segment -> elasticity estimates
        """
        elasticities = {}
        
        for segment in pricing_experiments['merchant_segment'].unique():
            segment_data = pricing_experiments[
                pricing_experiments['merchant_segment'] == segment
            ]
            
            if len(segment_data) < 3:
                elasticities[segment] = -0.5  # Default
                continue
            
            # Calculate elasticity from rate and conversion changes
            price_change = (
                segment_data['treatment_fee_rate'] - segment_data['control_fee_rate']
            ) / segment_data['control_fee_rate']
            
            quantity_change = (
                segment_data['treatment_sample_size'] - segment_data['control_sample_size']
            ) / segment_data['control_sample_size']
            
            # Avoid division by zero
            valid_mask = np.abs(price_change) > 0.001
            if valid_mask.sum() > 0:
                elasticity = (quantity_change[valid_mask] / price_change[valid_mask]).mean()
                elasticities[segment] = round(np.clip(elasticity, -2, 0), 3)
            else:
                elasticities[segment] = -0.5
        
        self.elasticity_estimates = elasticities
        return elasticities
    
    def predict_volume_impact(
        self, 
        segment: str,
        current_volume: float,
        price_change_pct: float
    ) -> Dict[str, float]:
        """
        Predict volume impact from price change.
        
        Args:
            segment: Merchant segment
            current_volume: Current transaction volume
            price_change_pct: Percentage price change
            
        Returns:
            Dictionary with predicted impact
        """
        elasticity = self.elasticity_estimates.get(segment, -0.5)
        
        volume_change_pct = price_change_pct * elasticity
        new_volume = current_volume * (1 + volume_change_pct / 100)
        
        return {
            'segment': segment,
            'elasticity': elasticity,
            'current_volume': current_volume,
            'price_change_pct': price_change_pct,
            'predicted_volume_change_pct': round(volume_change_pct, 2),
            'predicted_new_volume': round(new_volume, 2)
        }


def main():
    """Demo: Train and evaluate pricing optimization model."""
    # Import data generator
    import sys
    sys.path.insert(0, '../data')
    from generator import PayPalDataGenerator, DataConfig
    
    # Generate sample data
    config = DataConfig(
        num_merchants=2000,
        num_users=10000,
        num_transactions=100000
    )
    generator = PayPalDataGenerator(config)
    data = generator.generate_all()
    
    print("=" * 60)
    print("PRICING OPTIMIZATION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize and train model
    optimizer = PricingOptimizer()
    metrics = optimizer.fit(data['merchants'], data['transactions'])
    
    print(f"\nModel Performance:")
    print(f"  MAE: {metrics.mae:.4f} (${metrics.mae * 100:.2f} per $100)")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  R²: {metrics.r2:.4f}")
    print(f"  CV MAE: {metrics.cv_scores.mean():.4f} ± {metrics.cv_scores.std():.4f}")
    
    print(f"\nTop Feature Importances:")
    sorted_importance = sorted(
        metrics.feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")
    
    # Generate recommendations
    print("\n" + "=" * 60)
    print("PRICING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = optimizer.get_pricing_recommendations(
        data['merchants'], 
        data['transactions'],
        top_n=10
    )
    print(recommendations.to_string())
    
    # Elasticity analysis
    print("\n" + "=" * 60)
    print("PRICE ELASTICITY ANALYSIS")
    print("=" * 60)
    
    elasticity_analyzer = PriceElasticityAnalyzer()
    elasticities = elasticity_analyzer.estimate_elasticity(data['pricing_experiments'])
    
    print("\nElasticity by Segment:")
    for segment, elasticity in sorted(elasticities.items(), key=lambda x: x[1]):
        print(f"  {segment}: {elasticity}")
    
    # Simulate price change
    print("\n" + "=" * 60)
    print("PRICE CHANGE SIMULATION")
    print("=" * 60)
    
    sample_merchants = data['merchants'].head(100)
    current_rates = np.full(100, 0.029)
    new_rates = np.full(100, 0.025)  # 4% reduction
    volume = sample_merchants['monthly_volume_usd'].values
    
    simulation = optimizer.simulate_price_change(
        sample_merchants, current_rates, new_rates, volume
    )
    
    print("\nSimulation Results (100 merchants, -4% rate reduction):")
    for key, value in simulation.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
