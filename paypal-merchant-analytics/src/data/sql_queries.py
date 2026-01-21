"""
SQL Analytics Queries for PayPal Merchant Platform

Production-quality SQL queries for:
- Product performance dashboards
- Merchant behavior analysis
- Pricing optimization metrics
- Cross-border transaction analysis

Aligns with JD: "Use SQL, Python and Tableau to create product performance
dashboards that can be used by different teams across the company"
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    name: str
    description: str
    sql: str
    tableau_compatible: bool = True


class MerchantAnalyticsQueries:
    """
    SQL query library for merchant analytics.
    
    Designed for:
    - SQLite (development)
    - PostgreSQL (production)
    - Tableau connectivity
    """
    
    @staticmethod
    def monthly_transaction_volume() -> QueryResult:
        """
        Monthly transaction volume and revenue metrics.
        Used for: Leadership reporting, trend analysis
        """
        sql = """
        SELECT 
            strftime('%Y-%m', date) as month,
            COUNT(*) as total_transactions,
            COUNT(DISTINCT merchant_id) as active_merchants,
            COUNT(DISTINCT user_id) as active_users,
            ROUND(SUM(amount_usd), 2) as total_volume_usd,
            ROUND(AVG(amount_usd), 2) as avg_transaction_size,
            ROUND(SUM(total_fee_usd), 2) as total_fees_collected,
            ROUND(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as completion_rate,
            ROUND(SUM(CASE WHEN is_cross_border THEN amount_usd ELSE 0 END), 2) as cross_border_volume
        FROM transactions
        WHERE status IN ('completed', 'pending')
        GROUP BY strftime('%Y-%m', date)
        ORDER BY month DESC;
        """
        return QueryResult(
            name="monthly_transaction_volume",
            description="Monthly aggregated transaction metrics for executive dashboards",
            sql=sql
        )
    
    @staticmethod
    def merchant_performance_scorecard() -> QueryResult:
        """
        Comprehensive merchant performance metrics.
        Used for: Account management, risk assessment
        """
        sql = """
        WITH merchant_txn_stats AS (
            SELECT 
                merchant_id,
                COUNT(*) as total_transactions,
                SUM(amount_usd) as total_volume,
                AVG(amount_usd) as avg_transaction_size,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
                SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as refund_rate,
                SUM(CASE WHEN is_cross_border THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as cross_border_ratio,
                COUNT(DISTINCT date) as active_days,
                MIN(date) as first_transaction,
                MAX(date) as last_transaction
            FROM transactions
            GROUP BY merchant_id
        )
        SELECT 
            m.merchant_id,
            m.business_category,
            m.region,
            m.country,
            m.pricing_tier,
            m.account_age_months,
            m.churn_risk_score,
            COALESCE(t.total_transactions, 0) as total_transactions,
            ROUND(COALESCE(t.total_volume, 0), 2) as total_volume_usd,
            ROUND(COALESCE(t.avg_transaction_size, 0), 2) as avg_transaction_size,
            ROUND(COALESCE(t.success_rate, 0) * 100, 2) as success_rate_pct,
            ROUND(COALESCE(t.refund_rate, 0) * 100, 2) as refund_rate_pct,
            ROUND(COALESCE(t.cross_border_ratio, 0) * 100, 2) as cross_border_pct,
            COALESCE(t.active_days, 0) as active_days,
            t.first_transaction,
            t.last_transaction,
            -- Engagement score (composite metric)
            ROUND(
                (COALESCE(t.success_rate, 0) * 0.4 + 
                 (1 - COALESCE(t.refund_rate, 0)) * 0.3 +
                 LEAST(t.active_days / 365.0, 1) * 0.3) * 100, 
            2) as engagement_score
        FROM merchants m
        LEFT JOIN merchant_txn_stats t ON m.merchant_id = t.merchant_id
        ORDER BY total_volume_usd DESC;
        """
        return QueryResult(
            name="merchant_performance_scorecard",
            description="360-degree view of merchant performance for account management",
            sql=sql
        )
    
    @staticmethod
    def pricing_tier_analysis() -> QueryResult:
        """
        Analyze revenue and behavior by pricing tier.
        Used for: Pricing optimization, tier migration analysis
        """
        sql = """
        SELECT 
            m.pricing_tier,
            COUNT(DISTINCT m.merchant_id) as merchant_count,
            COUNT(t.transaction_id) as total_transactions,
            ROUND(SUM(t.amount_usd), 2) as total_volume_usd,
            ROUND(AVG(t.amount_usd), 2) as avg_transaction_size,
            ROUND(SUM(t.total_fee_usd), 2) as total_fees_collected,
            ROUND(AVG(t.fee_rate) * 100, 3) as avg_fee_rate_pct,
            ROUND(SUM(t.total_fee_usd) / NULLIF(SUM(t.amount_usd), 0) * 100, 3) as effective_take_rate_pct,
            ROUND(AVG(m.churn_risk_score), 4) as avg_churn_risk,
            -- Revenue per merchant
            ROUND(SUM(t.total_fee_usd) / NULLIF(COUNT(DISTINCT m.merchant_id), 0), 2) as avg_revenue_per_merchant
        FROM merchants m
        LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
        WHERE t.status = 'completed'
        GROUP BY m.pricing_tier
        ORDER BY total_fees_collected DESC;
        """
        return QueryResult(
            name="pricing_tier_analysis",
            description="Revenue and behavior metrics segmented by pricing tier",
            sql=sql
        )
    
    @staticmethod
    def cross_border_flow_analysis() -> QueryResult:
        """
        Analyze cross-border transaction patterns.
        Used for: International expansion strategy, corridor analysis
        """
        sql = """
        SELECT 
            merchant_country as origin_country,
            user_country as destination_country,
            COUNT(*) as transaction_count,
            ROUND(SUM(amount_usd), 2) as total_volume_usd,
            ROUND(AVG(amount_usd), 2) as avg_transaction_size,
            ROUND(SUM(total_fee_usd), 2) as total_fees,
            ROUND(AVG(fee_rate) * 100, 3) as avg_fee_rate_pct,
            -- Success metrics
            ROUND(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
        FROM transactions
        WHERE is_cross_border = 1
        GROUP BY merchant_country, user_country
        HAVING transaction_count >= 10
        ORDER BY total_volume_usd DESC
        LIMIT 50;
        """
        return QueryResult(
            name="cross_border_flow_analysis",
            description="Top cross-border transaction corridors by volume",
            sql=sql
        )
    
    @staticmethod
    def hourly_transaction_pattern() -> QueryResult:
        """
        Hourly transaction patterns for capacity planning.
        Used for: Infrastructure planning, peak hour analysis
        """
        sql = """
        SELECT 
            hour,
            day_of_week,
            COUNT(*) as transaction_count,
            ROUND(SUM(amount_usd), 2) as total_volume,
            ROUND(AVG(amount_usd), 2) as avg_amount,
            -- Peak indicator
            CASE 
                WHEN COUNT(*) > (SELECT AVG(cnt) * 1.5 FROM (
                    SELECT COUNT(*) as cnt FROM transactions GROUP BY hour, day_of_week
                )) THEN 'HIGH'
                WHEN COUNT(*) < (SELECT AVG(cnt) * 0.5 FROM (
                    SELECT COUNT(*) as cnt FROM transactions GROUP BY hour, day_of_week
                )) THEN 'LOW'
                ELSE 'NORMAL'
            END as traffic_level
        FROM transactions
        GROUP BY hour, day_of_week
        ORDER BY day_of_week, hour;
        """
        return QueryResult(
            name="hourly_transaction_pattern",
            description="Transaction distribution by hour and day for capacity planning",
            sql=sql
        )
    
    @staticmethod
    def category_performance() -> QueryResult:
        """
        Business category performance comparison.
        Used for: Vertical strategy, segment targeting
        """
        sql = """
        SELECT 
            m.business_category,
            COUNT(DISTINCT m.merchant_id) as merchant_count,
            COUNT(t.transaction_id) as total_transactions,
            ROUND(SUM(t.amount_usd), 2) as total_volume_usd,
            ROUND(AVG(t.amount_usd), 2) as avg_transaction_size,
            ROUND(SUM(t.total_fee_usd), 2) as total_fees,
            -- Category health metrics
            ROUND(AVG(m.churn_risk_score), 4) as avg_churn_risk,
            ROUND(AVG(m.dispute_rate) * 100, 3) as avg_dispute_rate_pct,
            ROUND(AVG(m.refund_rate) * 100, 3) as avg_refund_rate_pct,
            -- Growth indicator (last 30 days vs prior 30 days)
            ROUND(
                (SUM(CASE WHEN date >= date('now', '-30 days') THEN amount_usd ELSE 0 END) /
                 NULLIF(SUM(CASE WHEN date >= date('now', '-60 days') AND date < date('now', '-30 days') 
                            THEN amount_usd ELSE 0 END), 0) - 1) * 100,
            2) as mom_growth_pct
        FROM merchants m
        LEFT JOIN transactions t ON m.merchant_id = t.merchant_id AND t.status = 'completed'
        GROUP BY m.business_category
        ORDER BY total_volume_usd DESC;
        """
        return QueryResult(
            name="category_performance",
            description="Performance metrics by business vertical",
            sql=sql
        )
    
    @staticmethod
    def churn_risk_cohort() -> QueryResult:
        """
        Identify high-risk merchants for retention campaigns.
        Used for: Customer success, proactive retention
        """
        sql = """
        WITH recent_activity AS (
            SELECT 
                merchant_id,
                MAX(date) as last_transaction_date,
                COUNT(*) as recent_transactions,
                SUM(amount_usd) as recent_volume
            FROM transactions
            WHERE date >= date('now', '-90 days')
            GROUP BY merchant_id
        )
        SELECT 
            m.merchant_id,
            m.business_category,
            m.pricing_tier,
            m.account_age_months,
            m.monthly_volume_usd as expected_monthly_volume,
            m.churn_risk_score,
            COALESCE(r.last_transaction_date, 'No recent activity') as last_transaction_date,
            COALESCE(r.recent_transactions, 0) as transactions_last_90_days,
            COALESCE(r.recent_volume, 0) as volume_last_90_days,
            -- Risk classification
            CASE 
                WHEN m.churn_risk_score > 0.3 AND COALESCE(r.recent_transactions, 0) < 5 THEN 'CRITICAL'
                WHEN m.churn_risk_score > 0.2 OR COALESCE(r.recent_transactions, 0) < 10 THEN 'HIGH'
                WHEN m.churn_risk_score > 0.1 THEN 'MEDIUM'
                ELSE 'LOW'
            END as risk_tier,
            -- Estimated revenue at risk
            ROUND(m.monthly_volume_usd * m.churn_risk_score * 0.025, 2) as monthly_revenue_at_risk
        FROM merchants m
        LEFT JOIN recent_activity r ON m.merchant_id = r.merchant_id
        WHERE m.churn_risk_score > 0.1
        ORDER BY monthly_revenue_at_risk DESC
        LIMIT 1000;
        """
        return QueryResult(
            name="churn_risk_cohort",
            description="High-risk merchants ranked by revenue at risk",
            sql=sql
        )
    
    @staticmethod
    def platform_performance() -> QueryResult:
        """
        Transaction success rates by platform.
        Used for: Platform optimization, mobile strategy
        """
        sql = """
        SELECT 
            platform,
            COUNT(*) as total_transactions,
            ROUND(SUM(amount_usd), 2) as total_volume,
            ROUND(AVG(amount_usd), 2) as avg_transaction_size,
            -- Success metrics by status
            ROUND(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as completion_rate,
            ROUND(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as failure_rate,
            ROUND(SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as refund_rate,
            -- Payment method breakdown
            ROUND(SUM(CASE WHEN payment_method = 'card' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as card_pct,
            ROUND(SUM(CASE WHEN payment_method = 'balance' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as balance_pct
        FROM transactions
        GROUP BY platform
        ORDER BY total_volume DESC;
        """
        return QueryResult(
            name="platform_performance",
            description="Transaction metrics segmented by platform/channel",
            sql=sql
        )
    
    @staticmethod
    def fee_optimization_analysis() -> QueryResult:
        """
        Analyze fee structures for pricing optimization.
        Used for: Pricing strategy, competitive analysis
        """
        sql = """
        WITH fee_brackets AS (
            SELECT 
                CASE 
                    WHEN amount_usd < 10 THEN '<$10'
                    WHEN amount_usd < 50 THEN '$10-50'
                    WHEN amount_usd < 100 THEN '$50-100'
                    WHEN amount_usd < 500 THEN '$100-500'
                    WHEN amount_usd < 1000 THEN '$500-1000'
                    ELSE '$1000+'
                END as amount_bracket,
                amount_usd,
                total_fee_usd,
                fee_rate,
                fixed_fee,
                pricing_tier,
                is_cross_border
            FROM transactions
            WHERE status = 'completed'
        )
        SELECT 
            amount_bracket,
            pricing_tier,
            COUNT(*) as transaction_count,
            ROUND(AVG(amount_usd), 2) as avg_amount,
            ROUND(AVG(total_fee_usd), 2) as avg_fee,
            ROUND(AVG(total_fee_usd / amount_usd) * 100, 3) as effective_rate_pct,
            -- Fee burden analysis
            ROUND(AVG(CASE WHEN is_cross_border = 0 THEN total_fee_usd / amount_usd ELSE NULL END) * 100, 3) as domestic_rate_pct,
            ROUND(AVG(CASE WHEN is_cross_border = 1 THEN total_fee_usd / amount_usd ELSE NULL END) * 100, 3) as intl_rate_pct,
            -- Small merchant impact (high fee %)
            SUM(CASE WHEN total_fee_usd / amount_usd > 0.05 THEN 1 ELSE 0 END) as high_fee_burden_count
        FROM fee_brackets
        GROUP BY amount_bracket, pricing_tier
        ORDER BY pricing_tier, 
            CASE amount_bracket 
                WHEN '<$10' THEN 1 
                WHEN '$10-50' THEN 2 
                WHEN '$50-100' THEN 3 
                WHEN '$100-500' THEN 4 
                WHEN '$500-1000' THEN 5 
                ELSE 6 
            END;
        """
        return QueryResult(
            name="fee_optimization_analysis",
            description="Fee structure analysis by amount bracket for pricing optimization",
            sql=sql
        )
    
    @classmethod
    def get_all_queries(cls) -> Dict[str, QueryResult]:
        """Return all available queries as a dictionary."""
        return {
            'monthly_volume': cls.monthly_transaction_volume(),
            'merchant_scorecard': cls.merchant_performance_scorecard(),
            'pricing_tier': cls.pricing_tier_analysis(),
            'cross_border': cls.cross_border_flow_analysis(),
            'hourly_pattern': cls.hourly_transaction_pattern(),
            'category_performance': cls.category_performance(),
            'churn_risk': cls.churn_risk_cohort(),
            'platform_performance': cls.platform_performance(),
            'fee_optimization': cls.fee_optimization_analysis()
        }
    
    @classmethod
    def export_for_tableau(cls, output_path: str = 'dashboards/sql_queries.sql') -> None:
        """Export all queries to a SQL file for Tableau import."""
        queries = cls.get_all_queries()
        
        with open(output_path, 'w') as f:
            f.write("-- PayPal Merchant Analytics SQL Queries\n")
            f.write("-- Generated for Tableau Dashboard Integration\n")
            f.write("-- " + "=" * 60 + "\n\n")
            
            for name, query in queries.items():
                f.write(f"-- Query: {query.name}\n")
                f.write(f"-- Description: {query.description}\n")
                f.write(f"-- Tableau Compatible: {query.tableau_compatible}\n")
                f.write(query.sql)
                f.write("\n\n" + "-" * 60 + "\n\n")
        
        print(f"Exported {len(queries)} queries to {output_path}")


if __name__ == "__main__":
    # Export queries for documentation
    MerchantAnalyticsQueries.export_for_tableau()
    
    # Print query catalog
    print("Available Analytics Queries:")
    print("=" * 50)
    for name, query in MerchantAnalyticsQueries.get_all_queries().items():
        print(f"\n{name}:")
        print(f"  {query.description}")
