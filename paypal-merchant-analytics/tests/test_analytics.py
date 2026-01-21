"""
Test Suite for PayPal Merchant Analytics Platform

Comprehensive tests for:
- Data generation
- SQL queries
- ML models
- Visualization exports

Aligns with JD: "Develop production quality code with tests and documentation"
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'visualization'))


class TestDataGenerator:
    """Test suite for synthetic data generation."""
    
    @pytest.fixture
    def generator(self):
        """Create generator with small dataset for testing."""
        from generator import PayPalDataGenerator, DataConfig
        config = DataConfig(
            num_merchants=100,
            num_users=500,
            num_transactions=1000,
            seed=42
        )
        return PayPalDataGenerator(config)
    
    def test_merchant_generation(self, generator):
        """Test merchant data generation."""
        merchants = generator.generate_merchants()
        
        # Check shape
        assert len(merchants) == 100
        
        # Check required columns
        required_cols = [
            'merchant_id', 'business_category', 'region', 'country',
            'pricing_tier', 'monthly_volume_usd', 'churn_risk_score'
        ]
        for col in required_cols:
            assert col in merchants.columns
        
        # Check data quality
        assert merchants['merchant_id'].is_unique
        assert merchants['monthly_volume_usd'].min() > 0
        assert merchants['churn_risk_score'].between(0, 1).all()
        
        # Check categorical values
        valid_tiers = ['standard', 'preferred', 'enterprise', 'custom']
        assert merchants['pricing_tier'].isin(valid_tiers).all()
    
    def test_user_generation(self, generator):
        """Test user data generation."""
        users = generator.generate_users()
        
        assert len(users) == 500
        assert users['user_id'].is_unique
        
        valid_statuses = ['active', 'inactive', 'suspended']
        assert users['account_status'].isin(valid_statuses).all()
    
    def test_transaction_generation(self, generator):
        """Test transaction data generation."""
        generator.generate_merchants()
        generator.generate_users()
        transactions = generator.generate_transactions()
        
        assert len(transactions) == 1000
        assert transactions['transaction_id'].is_unique
        
        # Check numeric constraints
        assert transactions['amount_usd'].min() >= 1
        assert transactions['amount_usd'].max() <= 10000
        assert (transactions['total_fee_usd'] > 0).all()
        
        # Check foreign key relationships
        valid_merchants = set(generator.merchants_df['merchant_id'])
        valid_users = set(generator.users_df['user_id'])
        
        assert transactions['merchant_id'].isin(valid_merchants).all()
        assert transactions['user_id'].isin(valid_users).all()
        
        # Check status distribution
        valid_statuses = ['completed', 'pending', 'failed', 'refunded']
        assert transactions['status'].isin(valid_statuses).all()
    
    def test_fee_calculation(self, generator):
        """Test transaction fee calculations."""
        data = generator.generate_all()
        transactions = data['transactions']
        
        # Verify fee structure
        # Fee should be amount * rate + fixed fee
        expected_fee = (
            transactions['amount_usd'] * transactions['fee_rate'] + 
            transactions['fixed_fee']
        ).round(2)
        
        assert np.allclose(transactions['total_fee_usd'], expected_fee, rtol=0.01)
        
        # Net amount should be positive
        assert (transactions['net_amount_usd'] > 0).all()
    
    def test_reproducibility(self, generator):
        """Test that same seed produces same data."""
        from generator import PayPalDataGenerator, DataConfig
        
        config1 = DataConfig(num_merchants=50, seed=123)
        config2 = DataConfig(num_merchants=50, seed=123)
        
        gen1 = PayPalDataGenerator(config1)
        gen2 = PayPalDataGenerator(config2)
        
        merchants1 = gen1.generate_merchants()
        merchants2 = gen2.generate_merchants()
        
        pd.testing.assert_frame_equal(merchants1, merchants2)


class TestSQLQueries:
    """Test suite for SQL analytics queries."""
    
    @pytest.fixture
    def query_class(self):
        """Return query class."""
        from sql_queries import MerchantAnalyticsQueries
        return MerchantAnalyticsQueries
    
    def test_query_catalog(self, query_class):
        """Test that all queries are accessible."""
        queries = query_class.get_all_queries()
        
        assert len(queries) >= 5
        
        expected_queries = [
            'monthly_volume', 'merchant_scorecard', 'pricing_tier'
        ]
        for name in expected_queries:
            assert name in queries
    
    def test_query_structure(self, query_class):
        """Test query result structure."""
        queries = query_class.get_all_queries()
        
        for name, query_result in queries.items():
            assert hasattr(query_result, 'name')
            assert hasattr(query_result, 'sql')
            assert hasattr(query_result, 'description')
            assert len(query_result.sql) > 10  # Non-empty SQL
    
    def test_sql_syntax(self, query_class):
        """Basic SQL syntax validation."""
        queries = query_class.get_all_queries()
        
        for name, query_result in queries.items():
            sql = query_result.sql.upper()
            
            # Should have SELECT
            assert 'SELECT' in sql
            
            # Should have FROM
            assert 'FROM' in sql
            
            # Should not have obvious syntax errors
            assert sql.count('SELECT') >= 1


class TestPricingOptimizer:
    """Test suite for pricing optimization model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        from generator import PayPalDataGenerator, DataConfig
        config = DataConfig(
            num_merchants=500,
            num_users=1000,
            num_transactions=5000,
            seed=42
        )
        generator = PayPalDataGenerator(config)
        return generator.generate_all()
    
    @pytest.fixture
    def optimizer(self, sample_data):
        """Create and train optimizer."""
        from pricing_optimizer import PricingOptimizer
        opt = PricingOptimizer()
        opt.fit(sample_data['merchants'], sample_data['transactions'])
        return opt
    
    def test_model_fitting(self, sample_data):
        """Test that model fits without errors."""
        from pricing_optimizer import PricingOptimizer
        
        opt = PricingOptimizer()
        metrics = opt.fit(sample_data['merchants'], sample_data['transactions'])
        
        assert opt.is_fitted
        assert metrics.mae > 0
        assert metrics.r2 >= -1  # R2 can be negative for bad fits
    
    def test_prediction_shape(self, optimizer, sample_data):
        """Test prediction output shape."""
        predictions = optimizer.predict(sample_data['merchants'])
        
        assert len(predictions) == len(sample_data['merchants'])
        assert predictions.dtype == np.float64
    
    def test_prediction_bounds(self, optimizer, sample_data):
        """Test that predictions are within valid bounds."""
        predictions = optimizer.predict(sample_data['merchants'])
        
        # Fee rates should be between 1.5% and 4.5%
        assert predictions.min() >= 0.015
        assert predictions.max() <= 0.045
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        from pricing_optimizer import PricingOptimizer
        
        opt = PricingOptimizer()
        metrics = opt.fit(sample_data['merchants'], sample_data['transactions'])
        
        assert len(metrics.feature_importance) > 0
        
        # Importances should sum to approximately 1
        importance_sum = sum(metrics.feature_importance.values())
        assert 0.9 <= importance_sum <= 1.1
    
    def test_recommendations(self, optimizer, sample_data):
        """Test pricing recommendations generation."""
        recommendations = optimizer.get_pricing_recommendations(
            sample_data['merchants'],
            sample_data['transactions'],
            top_n=10
        )
        
        assert len(recommendations) == 10
        assert 'recommended_rate' in recommendations.columns
        assert 'revenue_impact' in recommendations.columns


class TestChurnPredictor:
    """Test suite for churn prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        from generator import PayPalDataGenerator, DataConfig
        config = DataConfig(
            num_merchants=500,
            num_users=1000,
            num_transactions=5000,
            seed=42
        )
        generator = PayPalDataGenerator(config)
        return generator.generate_all()
    
    def test_model_fitting(self, sample_data):
        """Test that churn model fits."""
        from churn_predictor import ChurnPredictor
        
        predictor = ChurnPredictor()
        metrics = predictor.fit(sample_data['merchants'], sample_data['transactions'])
        
        assert predictor.is_fitted
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.auc_roc <= 1
    
    def test_probability_output(self, sample_data):
        """Test probability predictions."""
        from churn_predictor import ChurnPredictor
        
        predictor = ChurnPredictor()
        predictor.fit(sample_data['merchants'], sample_data['transactions'])
        
        # Prepare test data
        from churn_predictor import ChurnPredictor
        test_data = predictor._prepare_churn_labels(
            sample_data['merchants'].head(50),
            sample_data['transactions']
        )
        
        probs = predictor.predict_proba(test_data)
        
        assert len(probs) == 50
        assert probs.min() >= 0
        assert probs.max() <= 1
    
    def test_risk_tier_assignment(self, sample_data):
        """Test risk tier assignment."""
        from churn_predictor import ChurnPredictor
        
        predictor = ChurnPredictor()
        predictor.fit(sample_data['merchants'], sample_data['transactions'])
        
        risk_tiers = predictor.assign_risk_tiers(
            sample_data['merchants'],
            sample_data['transactions']
        )
        
        valid_tiers = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        assert risk_tiers['risk_tier'].isin(valid_tiers).all()


class TestSentimentAnalyzer:
    """Test suite for sentiment analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer."""
        from sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer()
    
    def test_rule_based_positive(self, analyzer):
        """Test rule-based positive sentiment detection."""
        text = "Great service! The API integration was smooth and easy."
        result = analyzer.analyze_rule_based(text)
        
        assert result.sentiment == 'positive'
        assert result.confidence > 0.5
    
    def test_rule_based_negative(self, analyzer):
        """Test rule-based negative sentiment detection."""
        text = "Terrible experience. Fees are too high and support is slow."
        result = analyzer.analyze_rule_based(text)
        
        assert result.sentiment == 'negative'
        assert result.confidence > 0.5
    
    def test_rule_based_neutral(self, analyzer):
        """Test rule-based neutral sentiment detection."""
        text = "I have a question about the API documentation."
        result = analyzer.analyze_rule_based(text)
        
        assert result.sentiment == 'neutral'
    
    def test_category_detection(self, analyzer):
        """Test category detection."""
        billing_text = "The transaction fees and charges are confusing."
        result = analyzer.analyze_rule_based(billing_text)
        assert result.category == 'billing'
        
        tech_text = "The API webhook integration has errors."
        result = analyzer.analyze_rule_based(tech_text)
        assert result.category == 'technical'
    
    def test_ml_model_training(self):
        """Test ML model training."""
        from sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Sample training data
        texts = [
            "Great product!", "Excellent service",
            "Terrible experience", "Very disappointed",
            "Just okay", "Need more information"
        ] * 10
        labels = ['positive', 'positive', 'negative', 'negative', 'neutral', 'neutral'] * 10
        
        metrics = analyzer.fit(texts, labels)
        
        assert analyzer.is_fitted
        assert metrics.accuracy > 0


class TestDashboardGenerator:
    """Test suite for dashboard generation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        from generator import PayPalDataGenerator, DataConfig
        config = DataConfig(
            num_merchants=100,
            num_users=500,
            num_transactions=1000,
            seed=42
        )
        generator = PayPalDataGenerator(config)
        return generator.generate_all()
    
    def test_tableau_export(self, sample_data, tmp_path):
        """Test Tableau export functionality."""
        from dashboard_generator import TableauExporter, DashboardConfig
        
        config = DashboardConfig(output_dir=str(tmp_path))
        exporter = TableauExporter(config)
        
        filepath = exporter.export_transaction_summary(
            sample_data['transactions'],
            sample_data['merchants']
        )
        
        assert Path(filepath).exists()
        
        # Verify CSV is readable
        df = pd.read_csv(filepath)
        assert len(df) > 0
    
    def test_monthly_report(self, sample_data, tmp_path):
        """Test monthly report generation."""
        from dashboard_generator import MonthlyReportGenerator, DashboardConfig
        
        config = DashboardConfig(output_dir=str(tmp_path))
        reporter = MonthlyReportGenerator(config)
        
        metrics = reporter.generate_monthly_metrics(
            sample_data['transactions'],
            sample_data['merchants']
        )
        
        assert 'transaction_metrics' in metrics
        assert 'merchant_metrics' in metrics
        assert 'performance_metrics' in metrics
        
        assert metrics['transaction_metrics']['total_transactions'] > 0


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete data pipeline."""
        from generator import PayPalDataGenerator, DataConfig
        from pricing_optimizer import PricingOptimizer
        from churn_predictor import ChurnPredictor
        
        # Generate data
        config = DataConfig(
            num_merchants=200,
            num_users=1000,
            num_transactions=5000,
            seed=42
        )
        generator = PayPalDataGenerator(config)
        data = generator.generate_all()
        
        # Train pricing model
        pricing_opt = PricingOptimizer()
        pricing_metrics = pricing_opt.fit(data['merchants'], data['transactions'])
        assert pricing_opt.is_fitted
        
        # Train churn model
        churn_pred = ChurnPredictor()
        churn_metrics = churn_pred.fit(data['merchants'], data['transactions'])
        assert churn_pred.is_fitted
        
        # Generate predictions
        pricing_recommendations = pricing_opt.get_pricing_recommendations(
            data['merchants'], data['transactions'], top_n=20
        )
        assert len(pricing_recommendations) == 20
        
        risk_tiers = churn_pred.assign_risk_tiers(
            data['merchants'], data['transactions']
        )
        assert len(risk_tiers) == len(data['merchants'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
