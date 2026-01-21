"""
Merchant Churn Prediction Model

Binary classification model predicting merchant churn probability.
Uses behavioral features, transaction patterns, and engagement metrics.

Aligns with JD: "Build, under the supervision of senior data scientists 
and management team, machine learning models and perform data mining"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ChurnFeatures:
    """Feature configuration for churn model."""
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    target: str = 'churned'
    
    def __post_init__(self):
        if self.categorical_features is None:
            self.categorical_features = [
                'business_category', 'region', 'pricing_tier', 'integration_type'
            ]
        if self.numerical_features is None:
            self.numerical_features = [
                'account_age_months', 'monthly_volume_usd', 'avg_transaction_size',
                'dispute_rate', 'refund_rate', 'transaction_count',
                'days_since_last_txn', 'txn_frequency', 'avg_daily_volume',
                'cross_border_ratio', 'success_rate', 'volume_trend'
            ]


@dataclass
class ChurnMetrics:
    """Container for churn model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    cv_scores: np.ndarray


class ChurnPredictor:
    """
    Merchant churn prediction model.
    
    Capabilities:
    - Binary churn classification
    - Probability scoring
    - Risk tier assignment
    - Feature importance analysis
    - Retention campaign targeting
    """
    
    def __init__(self, feature_config: Optional[ChurnFeatures] = None):
        self.feature_config = feature_config or ChurnFeatures()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.threshold = 0.5
        
    def _prepare_churn_labels(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        churn_days: int = 90
    ) -> pd.DataFrame:
        """Prepare churn labels from transaction data."""
        max_date = pd.to_datetime(transactions_df['date']).max()
        
        last_txn = transactions_df.groupby('merchant_id').agg({
            'date': 'max',
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'status': lambda x: (x == 'completed').mean(),
            'is_cross_border': 'mean'
        }).reset_index()
        
        last_txn.columns = [
            'merchant_id', 'last_transaction_date', 'transaction_count',
            'total_volume', 'avg_transaction_size', 'success_rate', 
            'cross_border_ratio'
        ]
        
        last_txn['last_transaction_date'] = pd.to_datetime(last_txn['last_transaction_date'])
        last_txn['days_since_last_txn'] = (max_date - last_txn['last_transaction_date']).dt.days
        last_txn['churned'] = (last_txn['days_since_last_txn'] > churn_days).astype(int)
        
        date_range = (max_date - pd.to_datetime(transactions_df['date']).min()).days
        last_txn['txn_frequency'] = last_txn['transaction_count'] / max(date_range, 1)
        last_txn['avg_daily_volume'] = last_txn['total_volume'] / max(date_range, 1)
        
        # Volume trend
        recent_volume = transactions_df[
            pd.to_datetime(transactions_df['date']) > (max_date - pd.Timedelta(days=30))
        ].groupby('merchant_id')['amount_usd'].sum()
        
        prior_volume = transactions_df[
            (pd.to_datetime(transactions_df['date']) <= (max_date - pd.Timedelta(days=30))) &
            (pd.to_datetime(transactions_df['date']) > (max_date - pd.Timedelta(days=60)))
        ].groupby('merchant_id')['amount_usd'].sum()
        
        volume_trend = (recent_volume / prior_volume.replace(0, 1) - 1).fillna(0)
        last_txn['volume_trend'] = last_txn['merchant_id'].map(volume_trend).fillna(0)
        
        merged = merchants_df.merge(last_txn, on='merchant_id', how='left')
        merged['churned'] = merged['churned'].fillna(1)
        merged = merged.fillna(0)
        
        return merged
    
    def _prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """Prepare features for model training/prediction."""
        result = df.copy()
        
        for col in self.feature_config.categorical_features:
            if col in result.columns:
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    result[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        result[col].astype(str)
                    )
                elif col in self.label_encoders:
                    result[f'{col}_encoded'] = result[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0]
                        if str(x) in self.label_encoders[col].classes_ else -1
                    )
        
        return result
    
    def fit(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> ChurnMetrics:
        """Train the churn prediction model."""
        training_data = self._prepare_churn_labels(merchants_df, transactions_df)
        training_data = self._prepare_features(training_data, fit_encoders=True)
        
        feature_cols = (
            [f for f in self.feature_config.numerical_features if f in training_data.columns] +
            [f'{col}_encoded' for col in self.feature_config.categorical_features 
             if f'{col}_encoded' in training_data.columns]
        )
        
        self.feature_names = feature_cols
        
        X = training_data[feature_cols].fillna(0)
        y = training_data[self.feature_config.target]
        
        churn_rate = y.mean()
        print(f"Churn rate in training data: {churn_rate:.2%}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return ChurnMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_prob),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            feature_importance=importance,
            cv_scores=cv_scores
        )
    
    def predict_proba(self, merchant_data: pd.DataFrame) -> np.ndarray:
        """Predict churn probability for merchants."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        prepared = self._prepare_features(merchant_data, fit_encoders=False)
        available_features = [c for c in self.feature_names if c in prepared.columns]
        X = prepared[available_features].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, merchant_data: pd.DataFrame) -> np.ndarray:
        """Predict binary churn labels."""
        probabilities = self.predict_proba(merchant_data)
        return (probabilities >= self.threshold).astype(int)
    
    def assign_risk_tiers(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Assign risk tiers based on churn probability."""
        prepared = self._prepare_churn_labels(merchants_df, transactions_df)
        churn_probs = self.predict_proba(prepared)
        
        def get_tier(prob):
            if prob >= 0.7:
                return 'CRITICAL'
            elif prob >= 0.5:
                return 'HIGH'
            elif prob >= 0.3:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        prepared['churn_probability'] = churn_probs
        prepared['risk_tier'] = prepared['churn_probability'].apply(get_tier)
        prepared['monthly_revenue'] = prepared['monthly_volume_usd'] * 0.025
        prepared['revenue_at_risk'] = prepared['monthly_revenue'] * prepared['churn_probability']
        
        return prepared[[
            'merchant_id', 'business_category', 'pricing_tier',
            'monthly_volume_usd', 'days_since_last_txn', 'txn_frequency',
            'churn_probability', 'risk_tier', 'monthly_revenue', 'revenue_at_risk'
        ]].sort_values('churn_probability', ascending=False)
    
    def get_retention_targets(
        self,
        risk_tiers_df: pd.DataFrame,
        budget_usd: float = 100000,
        cost_per_intervention: float = 50
    ) -> pd.DataFrame:
        """Identify optimal merchants for retention campaigns."""
        actionable = risk_tiers_df[
            risk_tiers_df['risk_tier'].isin(['CRITICAL', 'HIGH', 'MEDIUM'])
        ].copy()
        
        effectiveness = {'CRITICAL': 0.3, 'HIGH': 0.4, 'MEDIUM': 0.5}
        
        actionable['intervention_effectiveness'] = actionable['risk_tier'].map(effectiveness)
        actionable['expected_value'] = (
            actionable['revenue_at_risk'] * 
            actionable['intervention_effectiveness'] * 12
        )
        actionable['roi'] = (actionable['expected_value'] - cost_per_intervention) / cost_per_intervention
        
        actionable = actionable.sort_values('roi', ascending=False)
        max_interventions = int(budget_usd / cost_per_intervention)
        targets = actionable.head(max_interventions)
        
        def get_recommendation(row):
            if row['risk_tier'] == 'CRITICAL':
                return "Account manager outreach + pricing review"
            elif row['risk_tier'] == 'HIGH':
                return "Automated re-engagement campaign"
            else:
                return "Feature adoption email series"
        
        targets['recommendation'] = targets.apply(get_recommendation, axis=1)
        
        return targets[[
            'merchant_id', 'business_category', 'risk_tier',
            'churn_probability', 'monthly_revenue', 'revenue_at_risk',
            'expected_value', 'roi', 'recommendation'
        ]]


def main():
    """Demo: Train and evaluate churn prediction model."""
    import sys
    sys.path.insert(0, '../data')
    from generator import PayPalDataGenerator, DataConfig
    
    config = DataConfig(num_merchants=3000, num_users=15000, num_transactions=150000)
    generator = PayPalDataGenerator(config)
    data = generator.generate_all()
    
    print("=" * 60)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    predictor = ChurnPredictor()
    metrics = predictor.fit(data['merchants'], data['transactions'])
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
    print(f"  F1 Score: {metrics.f1:.4f}")
    print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
    
    print(f"\nTop Feature Importances:")
    for feature, importance in sorted(metrics.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
