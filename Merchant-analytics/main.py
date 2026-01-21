#!/usr/bin/env python3
"""
PayPal Merchant Analytics Platform - Main Demo Script

Demonstrates the complete data science pipeline:
1. Synthetic data generation (mimicking PayPal data)
2. SQL analytics queries
3. ML model training (pricing optimization, churn prediction)
4. NLP sentiment analysis
5. Dashboard generation for Tableau

Run: python main.py
"""

import sys
from pathlib import Path

# Add source directories to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path / 'data'))
sys.path.insert(0, str(src_path / 'models'))
sys.path.insert(0, str(src_path / 'visualization'))

import pandas as pd
import numpy as np
from datetime import datetime


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_data_generation():
    """Generate synthetic PayPal-like data."""
    from generator import PayPalDataGenerator, DataConfig
    
    print_header("1. SYNTHETIC DATA GENERATION")
    
    config = DataConfig(
        num_merchants=3000,
        num_users=15000,
        num_transactions=100000,
        start_date="2023-01-01",
        end_date="2024-12-31",
        seed=42
    )
    
    print(f"\nConfiguration:")
    print(f"  Merchants: {config.num_merchants:,}")
    print(f"  Users: {config.num_users:,}")
    print(f"  Transactions: {config.num_transactions:,}")
    
    generator = PayPalDataGenerator(config)
    data = generator.generate_all()
    
    print(f"\nGenerated Data Summary:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} records, {len(df.columns)} columns")
    
    # Save to CSV
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(output_dir / f'{name}.csv', index=False)
    
    print(f"\nData saved to: {output_dir}")
    return data


def run_sql_analytics(data: dict):
    """Run SQL analytics queries."""
    from sql_queries import MerchantAnalyticsQueries
    
    print_header("2. SQL ANALYTICS QUERIES")
    
    queries = MerchantAnalyticsQueries.get_all_queries()
    
    print(f"\nAvailable Queries ({len(queries)}):")
    for name, query_result in queries.items():
        print(f"  • {name}: {query_result.description[:55]}...")
    
    output_path = Path('dashboards')
    output_path.mkdir(parents=True, exist_ok=True)
    MerchantAnalyticsQueries.export_for_tableau(str(output_path / 'sql_queries.sql'))
    
    return queries


def run_pricing_optimization(data: dict):
    """Train and evaluate pricing optimization model."""
    from pricing_optimizer import PricingOptimizer, PriceElasticityAnalyzer
    
    print_header("3. PRICING OPTIMIZATION MODEL")
    
    optimizer = PricingOptimizer()
    metrics = optimizer.fit(data['merchants'], data['transactions'])
    
    print(f"\nModel Performance:")
    print(f"  Mean Absolute Error: {metrics.mae:.4f} ({metrics.mae * 100:.2f}% error)")
    print(f"  R² Score: {metrics.r2:.4f}")
    print(f"  CV MAE: {metrics.cv_scores.mean():.4f} ± {metrics.cv_scores.std():.4f}")
    
    print(f"\nTop 5 Feature Importances:")
    sorted_imp = sorted(metrics.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in sorted_imp:
        bar = "█" * int(importance * 50)
        print(f"  {feature:30s} {importance:.3f} {bar}")
    
    recommendations = optimizer.get_pricing_recommendations(
        data['merchants'], data['transactions'], top_n=10
    )
    
    print(f"\nTop 10 Pricing Recommendations:")
    print(recommendations[['merchant_id', 'pricing_tier', 'current_rate', 
                          'recommended_rate', 'revenue_impact']].to_string(index=False))
    
    return optimizer, metrics


def run_churn_prediction(data: dict):
    """Train and evaluate churn prediction model."""
    from churn_predictor import ChurnPredictor
    
    print_header("4. CHURN PREDICTION MODEL")
    
    predictor = ChurnPredictor()
    metrics = predictor.fit(data['merchants'], data['transactions'])
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
    print(f"  F1 Score: {metrics.f1:.4f}")
    print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
    
    risk_tiers = predictor.assign_risk_tiers(data['merchants'], data['transactions'])
    
    print(f"\nRisk Tier Distribution:")
    tier_dist = risk_tiers['risk_tier'].value_counts()
    for tier, count in tier_dist.items():
        pct = count / len(risk_tiers) * 100
        bar = "█" * int(pct / 2)
        print(f"  {tier:10s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    print(f"\nRevenue at Risk by Tier:")
    revenue_risk = risk_tiers.groupby('risk_tier')['revenue_at_risk'].sum()
    for tier, revenue in revenue_risk.sort_values(ascending=False).items():
        print(f"  {tier}: ${revenue:,.2f}")
    
    targets = predictor.get_retention_targets(risk_tiers, budget_usd=50000)
    print(f"\nRetention Campaign (Budget: $50,000):")
    print(f"  Merchants targeted: {len(targets):,}")
    print(f"  Expected value: ${targets['expected_value'].sum():,.2f}")
    
    return predictor, risk_tiers


def run_sentiment_analysis(data: dict):
    """Train and evaluate sentiment analysis model."""
    from sentiment_analyzer import SentimentAnalyzer, FeedbackCategorizer
    
    print_header("5. NLP SENTIMENT ANALYSIS")
    
    analyzer = SentimentAnalyzer()
    texts = data['feedback']['text'].tolist()
    labels = data['feedback']['sentiment_label'].tolist()
    
    metrics = analyzer.fit(texts, labels)
    print(f"\nML Model Accuracy: {metrics.accuracy:.4f}")
    
    print(f"\n--- Sample Feedback Analysis ---")
    test_samples = [
        "The checkout experience is amazing! Conversion rates improved.",
        "Transaction fees are too high. Considering switching.",
        "Looking for API documentation on webhook implementation.",
    ]
    
    for text in test_samples:
        result = analyzer.analyze(text, method='hybrid')
        print(f"\nText: '{text[:45]}...'")
        print(f"  Sentiment: {result.sentiment} ({result.confidence:.0%})")
        print(f"  Category: {result.category}")
    
    return analyzer


def run_dashboard_generation(data: dict, risk_tiers):
    """Generate dashboards and exports."""
    from dashboard_generator import TableauExporter, MonthlyReportGenerator, DashboardConfig
    
    print_header("6. DASHBOARD & REPORT GENERATION")
    
    config = DashboardConfig(output_dir='dashboards')
    
    print("\nGenerating Tableau-compatible exports...")
    exporter = TableauExporter(config)
    
    exports = [
        exporter.export_transaction_summary(data['transactions'], data['merchants']),
        exporter.export_merchant_scorecard(data['merchants'], data['transactions']),
        exporter.export_pricing_analysis(data['transactions']),
        exporter.export_cross_border_flows(data['transactions'])
    ]
    
    print(f"\nTableau Exports Created:")
    for filepath in exports:
        print(f"  • {filepath}")
    
    print("\nGenerating monthly report...")
    reporter = MonthlyReportGenerator(config)
    metrics = reporter.generate_monthly_metrics(data['transactions'], data['merchants'])
    
    print(f"\n--- Monthly Executive Summary ---")
    print(f"Report Period: {metrics['report_month']}")
    print(f"  Total Transactions: {metrics['transaction_metrics']['total_transactions']:,}")
    print(f"  Total Volume: ${metrics['transaction_metrics']['total_volume_usd']:,.2f}")
    print(f"  Total Fees: ${metrics['transaction_metrics']['total_fees_usd']:,.2f}")
    print(f"  Active Merchants: {metrics['merchant_metrics']['active_merchants']:,}")
    
    return exports


def main():
    """Run complete analytics pipeline."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PayPal Merchant Analytics Platform" + " " * 18 + "║")
    print("║" + " " * 20 + "Data Science Portfolio Project" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    
    start_time = datetime.now()
    
    # Run pipeline
    data = run_data_generation()
    queries = run_sql_analytics(data)
    pricing_model, pricing_metrics = run_pricing_optimization(data)
    churn_model, risk_tiers = run_churn_prediction(data)
    sentiment_model = run_sentiment_analysis(data)
    exports = run_dashboard_generation(data, risk_tiers)
    
    # Summary
    print_header("PIPELINE COMPLETE")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✅ Pipeline executed successfully in {elapsed:.1f} seconds")
    print(f"\nArtifacts Generated:")
    print(f"  • Data files: data/raw/")
    print(f"  • SQL queries: dashboards/sql_queries.sql")
    print(f"  • Tableau exports: dashboards/tableau_exports/")
    print(f"  • Monthly report: dashboards/reports/")
    
    print(f"\nModel Performance Summary:")
    print(f"  • Pricing Optimization MAE: {pricing_metrics.mae:.4f}")
    print(f"  • Pricing Optimization R²: {pricing_metrics.r2:.4f}")
    
    print(f"\n" + "-" * 70)
    print("Ready for presentation to PayPal Data Science team!")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
