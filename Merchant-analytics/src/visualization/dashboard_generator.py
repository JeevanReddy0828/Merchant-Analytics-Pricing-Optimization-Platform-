"""
Dashboard Generator for Product Performance Visualization

Creates Tableau-ready exports and interactive visualizations.
Supports automated monthly reporting.

Aligns with JD: "Use SQL, Python and Tableau to create product performance
dashboards that can be used by different teams across the company"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available. Install with: pip install plotly")


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    output_dir: str = 'dashboards'
    date_format: str = '%Y-%m-%d'
    theme: str = 'plotly_white'
    width: int = 1200
    height: int = 600


class TableauExporter:
    """
    Export data in Tableau-compatible formats.
    
    Features:
    - CSV exports with proper formatting
    - Metadata files for Tableau import
    - Data dictionary generation
    - Refresh automation support
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize exporter with configuration."""
        self.config = config or DashboardConfig()
        self.output_path = Path(self.config.output_dir) / 'tableau_exports'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def export_transaction_summary(
        self, 
        transactions_df: pd.DataFrame,
        merchants_df: pd.DataFrame
    ) -> str:
        """
        Export transaction summary for Tableau.
        
        Creates aggregated data suitable for:
        - Time series analysis
        - Geographic breakdown
        - Category comparison
        
        Args:
            transactions_df: Transaction records
            merchants_df: Merchant profiles
            
        Returns:
            Path to exported file
        """
        # Merge with merchant data
        merged = transactions_df.merge(
            merchants_df[['merchant_id', 'business_category', 'region', 'pricing_tier']],
            on='merchant_id',
            how='left',
            suffixes=('', '_merchant')
        )
        
        # Use business_category from transactions if available, otherwise from merge
        if 'business_category' not in merged.columns and 'business_category_merchant' in merged.columns:
            merged['business_category'] = merged['business_category_merchant']
        
        # Daily aggregation
        daily_summary = merged.groupby(['date', 'business_category', 'region']).agg({
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'total_fee_usd': 'sum',
            'is_cross_border': 'sum'
        }).reset_index()
        
        daily_summary.columns = [
            'date', 'business_category', 'region',
            'transaction_count', 'total_volume', 'avg_transaction_size',
            'total_fees', 'cross_border_count'
        ]
        
        # Add calculated fields for Tableau
        daily_summary['take_rate'] = daily_summary['total_fees'] / daily_summary['total_volume']
        daily_summary['cross_border_ratio'] = daily_summary['cross_border_count'] / daily_summary['transaction_count']
        
        # Export
        filepath = self.output_path / 'daily_transaction_summary.csv'
        daily_summary.to_csv(filepath, index=False)
        
        # Create data dictionary
        self._create_data_dictionary(daily_summary, 'daily_transaction_summary')
        
        return str(filepath)
    
    def export_merchant_scorecard(
        self, 
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> str:
        """
        Export merchant scorecard data.
        
        Args:
            merchants_df: Merchant profiles
            transactions_df: Transaction records
            
        Returns:
            Path to exported file
        """
        # Aggregate transaction metrics per merchant
        txn_metrics = transactions_df.groupby('merchant_id').agg({
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'total_fee_usd': 'sum',
            'status': lambda x: (x == 'completed').mean(),
            'is_cross_border': 'mean'
        }).reset_index()
        
        txn_metrics.columns = [
            'merchant_id', 'transaction_count', 'total_volume',
            'avg_transaction_size', 'total_fees', 'success_rate',
            'cross_border_ratio'
        ]
        
        # Merge
        scorecard = merchants_df.merge(txn_metrics, on='merchant_id', how='left')
        scorecard = scorecard.fillna(0)
        
        # Add risk tier
        def get_risk_tier(row):
            if row['churn_risk_score'] > 0.3:
                return 'High Risk'
            elif row['churn_risk_score'] > 0.15:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        scorecard['risk_tier'] = scorecard.apply(get_risk_tier, axis=1)
        
        # Export
        filepath = self.output_path / 'merchant_scorecard.csv'
        scorecard.to_csv(filepath, index=False)
        
        self._create_data_dictionary(scorecard, 'merchant_scorecard')
        
        return str(filepath)
    
    def export_pricing_analysis(
        self, 
        transactions_df: pd.DataFrame
    ) -> str:
        """
        Export pricing analysis data.
        
        Args:
            transactions_df: Transaction records
            
        Returns:
            Path to exported file
        """
        # Fee analysis by amount bucket
        transactions_df['amount_bucket'] = pd.cut(
            transactions_df['amount_usd'],
            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
            labels=['<$10', '$10-50', '$50-100', '$100-500', '$500-1000', '$1000+']
        )
        
        pricing_analysis = transactions_df.groupby(
            ['amount_bucket', 'pricing_tier', 'is_cross_border']
        ).agg({
            'transaction_id': 'count',
            'amount_usd': ['sum', 'mean'],
            'total_fee_usd': ['sum', 'mean'],
            'fee_rate': 'mean'
        }).reset_index()
        
        pricing_analysis.columns = [
            'amount_bucket', 'pricing_tier', 'is_cross_border',
            'transaction_count', 'total_volume', 'avg_amount',
            'total_fees', 'avg_fee', 'avg_fee_rate'
        ]
        
        pricing_analysis['effective_rate'] = pricing_analysis['total_fees'] / pricing_analysis['total_volume']
        
        filepath = self.output_path / 'pricing_analysis.csv'
        pricing_analysis.to_csv(filepath, index=False)
        
        self._create_data_dictionary(pricing_analysis, 'pricing_analysis')
        
        return str(filepath)
    
    def export_cross_border_flows(
        self, 
        transactions_df: pd.DataFrame
    ) -> str:
        """
        Export cross-border transaction flows.
        
        Args:
            transactions_df: Transaction records
            
        Returns:
            Path to exported file
        """
        # Filter to cross-border
        xb_txns = transactions_df[transactions_df['is_cross_border'] == True].copy()
        
        # Flow analysis
        flows = xb_txns.groupby(['merchant_country', 'user_country']).agg({
            'transaction_id': 'count',
            'amount_usd': 'sum',
            'total_fee_usd': 'sum'
        }).reset_index()
        
        flows.columns = ['origin_country', 'destination_country', 
                         'transaction_count', 'volume', 'fees']
        
        # Add flow metrics
        flows['avg_transaction_size'] = flows['volume'] / flows['transaction_count']
        flows['effective_rate'] = flows['fees'] / flows['volume']
        
        filepath = self.output_path / 'cross_border_flows.csv'
        flows.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def _create_data_dictionary(self, df: pd.DataFrame, name: str) -> None:
        """
        Create data dictionary for Tableau users.
        
        Args:
            df: DataFrame to document
            name: Dataset name
        """
        dictionary = {
            'dataset_name': name,
            'generated_at': datetime.now().isoformat(),
            'row_count': len(df),
            'columns': []
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].notna().sum()),
                'unique_values': int(df[col].nunique())
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info['min'] = float(df[col].min())
                col_info['max'] = float(df[col].max())
                col_info['mean'] = float(df[col].mean())
            
            dictionary['columns'].append(col_info)
        
        dict_path = self.output_path / f'{name}_dictionary.json'
        with open(dict_path, 'w') as f:
            json.dump(dictionary, f, indent=2)


class InteractiveDashboard:
    """
    Generate interactive visualizations using Plotly.
    
    Creates standalone HTML dashboards that can be:
    - Shared with stakeholders
    - Embedded in reports
    - Used for presentations
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard generator."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly required. Install with: pip install plotly")
        
        self.config = config or DashboardConfig()
        self.output_path = Path(self.config.output_dir) / 'interactive'
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_executive_dashboard(
        self,
        transactions_df: pd.DataFrame,
        merchants_df: pd.DataFrame
    ) -> str:
        """
        Create executive summary dashboard.
        
        Includes:
        - KPI cards
        - Volume trend
        - Category breakdown
        - Regional distribution
        
        Args:
            transactions_df: Transaction records
            merchants_df: Merchant profiles
            
        Returns:
            Path to HTML file
        """
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Transaction Volume',
                'Revenue by Business Category',
                'Transaction Volume by Region',
                'Merchant Risk Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # 1. Monthly volume trend
        monthly = transactions_df.groupby(
            pd.to_datetime(transactions_df['date']).dt.to_period('M')
        ).agg({
            'amount_usd': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        monthly['date'] = monthly['date'].astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=monthly['date'],
                y=monthly['amount_usd'],
                mode='lines+markers',
                name='Volume (USD)',
                line=dict(color='#003087')
            ),
            row=1, col=1
        )
        
        # 2. Revenue by category
        merged = transactions_df.merge(
            merchants_df[['merchant_id', 'business_category']],
            on='merchant_id'
        )
        category_revenue = merged.groupby('business_category')['total_fee_usd'].sum().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=category_revenue.values,
                y=category_revenue.index,
                orientation='h',
                marker_color='#0070ba',
                name='Revenue'
            ),
            row=1, col=2
        )
        
        # 3. Regional distribution (pie)
        regional = merged.groupby('merchant_country')['amount_usd'].sum().nlargest(10)
        
        fig.add_trace(
            go.Pie(
                labels=regional.index,
                values=regional.values,
                hole=0.4,
                name='Regional'
            ),
            row=2, col=1
        )
        
        # 4. Risk distribution
        risk_dist = pd.cut(
            merchants_df['churn_risk_score'],
            bins=[0, 0.15, 0.3, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        ).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=risk_dist.index.astype(str),
                y=risk_dist.values,
                marker_color=['#28a745', '#ffc107', '#dc3545'],
                name='Merchants'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='PayPal Merchant Analytics - Executive Dashboard',
            height=800,
            showlegend=False,
            template=self.config.theme
        )
        
        filepath = self.output_path / 'executive_dashboard.html'
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def create_pricing_dashboard(
        self,
        transactions_df: pd.DataFrame,
        pricing_experiments: pd.DataFrame
    ) -> str:
        """
        Create pricing analysis dashboard.
        
        Includes:
        - Fee distribution
        - Price elasticity curves
        - Experiment results
        - Tier comparison
        
        Args:
            transactions_df: Transaction records
            pricing_experiments: A/B test results
            
        Returns:
            Path to HTML file
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Fee Distribution by Transaction Size',
                'Effective Rate by Pricing Tier',
                'A/B Test Results',
                'Revenue Impact Analysis'
            )
        )
        
        # 1. Fee distribution histogram
        fig.add_trace(
            go.Histogram(
                x=transactions_df['total_fee_usd'],
                nbinsx=50,
                marker_color='#003087',
                name='Fee Distribution'
            ),
            row=1, col=1
        )
        
        # 2. Effective rate by tier
        tier_rates = transactions_df.groupby('pricing_tier').apply(
            lambda x: (x['total_fee_usd'].sum() / x['amount_usd'].sum()) * 100
        ).sort_values()
        
        fig.add_trace(
            go.Bar(
                x=tier_rates.index,
                y=tier_rates.values,
                marker_color='#0070ba',
                name='Effective Rate %'
            ),
            row=1, col=2
        )
        
        # 3. A/B test results
        fig.add_trace(
            go.Scatter(
                x=pricing_experiments['control_revenue'],
                y=pricing_experiments['treatment_revenue'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=pricing_experiments['is_significant'].astype(int),
                    colorscale=['red', 'green'],
                    showscale=True
                ),
                name='Experiments'
            ),
            row=2, col=1
        )
        
        # 4. Revenue by category
        category_revenue = transactions_df.groupby('business_category')['total_fee_usd'].sum().nlargest(10)
        
        fig.add_trace(
            go.Bar(
                x=category_revenue.index,
                y=category_revenue.values,
                marker_color='#28a745',
                name='Revenue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Pricing Optimization Dashboard',
            height=800,
            showlegend=False,
            template=self.config.theme
        )
        
        filepath = self.output_path / 'pricing_dashboard.html'
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def create_merchant_health_dashboard(
        self,
        merchants_df: pd.DataFrame,
        risk_tiers_df: pd.DataFrame
    ) -> str:
        """
        Create merchant health monitoring dashboard.
        
        Args:
            merchants_df: Merchant profiles
            risk_tiers_df: Risk tier assignments
            
        Returns:
            Path to HTML file
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Churn Risk Distribution',
                'Risk by Business Category',
                'Revenue at Risk by Tier',
                'Merchant Activity Heatmap'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Churn risk histogram
        fig.add_trace(
            go.Histogram(
                x=merchants_df['churn_risk_score'],
                nbinsx=30,
                marker_color='#dc3545',
                name='Risk Score'
            ),
            row=1, col=1
        )
        
        # 2. Risk by category
        if 'churn_probability' in risk_tiers_df.columns:
            category_risk = risk_tiers_df.groupby('business_category')['churn_probability'].mean().sort_values()
            
            fig.add_trace(
                go.Bar(
                    x=category_risk.index,
                    y=category_risk.values,
                    marker_color='#ffc107',
                    name='Avg Risk'
                ),
                row=1, col=2
            )
        
        # 3. Revenue at risk
        if 'revenue_at_risk' in risk_tiers_df.columns:
            tier_risk = risk_tiers_df.groupby('risk_tier')['revenue_at_risk'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=tier_risk.index,
                    y=tier_risk.values,
                    marker_color=['#28a745', '#ffc107', '#fd7e14', '#dc3545'],
                    name='Revenue at Risk'
                ),
                row=2, col=1
            )
        
        # 4. Activity heatmap (volume by tier and category)
        pivot = merchants_df.pivot_table(
            values='monthly_volume_usd',
            index='pricing_tier',
            columns='business_category',
            aggfunc='sum'
        ).fillna(0)
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Blues',
                name='Volume'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Merchant Health Dashboard',
            height=800,
            showlegend=False,
            template=self.config.theme
        )
        
        filepath = self.output_path / 'merchant_health_dashboard.html'
        fig.write_html(str(filepath))
        
        return str(filepath)


class MonthlyReportGenerator:
    """
    Automated monthly report generation.
    
    Creates comprehensive reports for leadership presentations.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize report generator."""
        self.config = config or DashboardConfig()
        self.output_path = Path(self.config.output_dir) / 'reports'
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def generate_monthly_metrics(
        self,
        transactions_df: pd.DataFrame,
        merchants_df: pd.DataFrame,
        report_month: Optional[str] = None
    ) -> Dict:
        """
        Generate monthly metrics summary.
        
        Args:
            transactions_df: Transaction records
            merchants_df: Merchant profiles
            report_month: Month to report (YYYY-MM), defaults to latest
            
        Returns:
            Dictionary with monthly metrics
        """
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
        
        if report_month is None:
            report_month = transactions_df['month'].max()
        else:
            report_month = pd.Period(report_month)
        
        current_month = transactions_df[transactions_df['month'] == report_month]
        prior_month = transactions_df[transactions_df['month'] == report_month - 1]
        
        def safe_pct_change(current, prior):
            if prior == 0:
                return 0
            return ((current - prior) / prior) * 100
        
        metrics = {
            'report_month': str(report_month),
            'generated_at': datetime.now().isoformat(),
            'transaction_metrics': {
                'total_transactions': int(len(current_month)),
                'total_volume_usd': round(current_month['amount_usd'].sum(), 2),
                'total_fees_usd': round(current_month['total_fee_usd'].sum(), 2),
                'avg_transaction_size': round(current_month['amount_usd'].mean(), 2),
                'mom_transaction_change': round(
                    safe_pct_change(len(current_month), len(prior_month)), 2
                ),
                'mom_volume_change': round(
                    safe_pct_change(
                        current_month['amount_usd'].sum(),
                        prior_month['amount_usd'].sum()
                    ), 2
                )
            },
            'merchant_metrics': {
                'total_merchants': int(len(merchants_df)),
                'active_merchants': int(current_month['merchant_id'].nunique()),
                'new_merchants': int(len(merchants_df[merchants_df['account_age_months'] <= 1])),
                'high_risk_merchants': int(len(merchants_df[merchants_df['churn_risk_score'] > 0.3]))
            },
            'performance_metrics': {
                'success_rate': round(
                    (current_month['status'] == 'completed').mean() * 100, 2
                ),
                'cross_border_ratio': round(
                    current_month['is_cross_border'].mean() * 100, 2
                ),
                'effective_take_rate': round(
                    current_month['total_fee_usd'].sum() / 
                    current_month['amount_usd'].sum() * 100, 3
                )
            }
        }
        
        # Save report
        report_path = self.output_path / f'monthly_report_{report_month}.json'
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """Demo: Generate all dashboards and exports."""
    import sys
    sys.path.insert(0, '../data')
    from generator import PayPalDataGenerator, DataConfig
    
    # Generate sample data
    config = DataConfig(num_merchants=3000, num_users=15000, num_transactions=150000)
    generator = PayPalDataGenerator(config)
    data = generator.generate_all()
    
    print("=" * 60)
    print("DASHBOARD GENERATION")
    print("=" * 60)
    
    # Tableau exports
    print("\n1. Generating Tableau Exports...")
    exporter = TableauExporter()
    
    txn_export = exporter.export_transaction_summary(data['transactions'], data['merchants'])
    print(f"   Transaction summary: {txn_export}")
    
    scorecard_export = exporter.export_merchant_scorecard(data['merchants'], data['transactions'])
    print(f"   Merchant scorecard: {scorecard_export}")
    
    pricing_export = exporter.export_pricing_analysis(data['transactions'])
    print(f"   Pricing analysis: {pricing_export}")
    
    # Interactive dashboards
    if HAS_PLOTLY:
        print("\n2. Generating Interactive Dashboards...")
        dashboard = InteractiveDashboard()
        
        exec_dashboard = dashboard.create_executive_dashboard(
            data['transactions'], data['merchants']
        )
        print(f"   Executive dashboard: {exec_dashboard}")
        
        pricing_dashboard = dashboard.create_pricing_dashboard(
            data['transactions'], data['pricing_experiments']
        )
        print(f"   Pricing dashboard: {pricing_dashboard}")
    
    # Monthly report
    print("\n3. Generating Monthly Report...")
    reporter = MonthlyReportGenerator()
    metrics = reporter.generate_monthly_metrics(data['transactions'], data['merchants'])
    
    print(f"\nMonthly Metrics Summary:")
    print(f"  Total Transactions: {metrics['transaction_metrics']['total_transactions']:,}")
    print(f"  Total Volume: ${metrics['transaction_metrics']['total_volume_usd']:,.2f}")
    print(f"  Total Fees: ${metrics['transaction_metrics']['total_fees_usd']:,.2f}")
    print(f"  Active Merchants: {metrics['merchant_metrics']['active_merchants']:,}")
    print(f"  Success Rate: {metrics['performance_metrics']['success_rate']}%")
    
    print("\n✅ All dashboards generated successfully!")


if __name__ == "__main__":
    main()
