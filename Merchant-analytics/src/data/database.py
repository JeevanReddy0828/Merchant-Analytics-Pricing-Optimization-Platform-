"""
Database Operations for PayPal Merchant Analytics

Supports both SQLite (development) and PostgreSQL (production).
Implements data loading, query execution, and result export.

Aligns with JD: "Implement large-scale data mining and machine learning 
algorithms and data model pipelines in a software production environment"
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
import logging

from sql_queries import MerchantAnalyticsQueries, QueryResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database abstraction layer for analytics operations.
    
    Features:
    - SQLite for local development
    - PostgreSQL-compatible query syntax
    - Batch data loading
    - Query result export for Tableau
    """
    
    def __init__(self, db_path: str = "data/analytics.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        
    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.rowcount
    
    def load_csv_to_table(
        self, 
        csv_path: str, 
        table_name: str, 
        if_exists: str = 'replace'
    ) -> int:
        """
        Load CSV file into database table.
        
        Args:
            csv_path: Path to CSV file
            table_name: Target table name
            if_exists: How to handle existing table ('replace', 'append', 'fail')
            
        Returns:
            Number of rows loaded
        """
        df = pd.read_csv(csv_path)
        
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logger.info(f"Loaded {len(df)} rows into {table_name}")
            
        return len(df)
    
    def load_all_data(self, data_dir: str = 'data/raw') -> Dict[str, int]:
        """
        Load all CSV files from data directory into database.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            Dictionary of table names to row counts
        """
        data_path = Path(data_dir)
        results = {}
        
        for csv_file in data_path.glob('*.csv'):
            table_name = csv_file.stem
            rows = self.load_csv_to_table(str(csv_file), table_name)
            results[table_name] = rows
            
        return results
    
    def create_indexes(self) -> None:
        """Create indexes for optimal query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant_id)",
            "CREATE INDEX IF NOT EXISTS idx_txn_user ON transactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date)",
            "CREATE INDEX IF NOT EXISTS idx_txn_status ON transactions(status)",
            "CREATE INDEX IF NOT EXISTS idx_merchant_tier ON merchants(pricing_tier)",
            "CREATE INDEX IF NOT EXISTS idx_merchant_category ON merchants(business_category)",
            "CREATE INDEX IF NOT EXISTS idx_merchant_churn ON merchants(churn_risk_score)",
        ]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for idx_sql in indexes:
                cursor.execute(idx_sql)
            conn.commit()
            
        logger.info(f"Created {len(indexes)} indexes")
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table."""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get basic statistics for a table."""
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        count_result = self.execute_query(count_query)
        
        return {
            'table_name': table_name,
            'row_count': count_result['row_count'].iloc[0],
            'columns': list(self.get_table_info(table_name)['name'])
        }


class AnalyticsQueryExecutor:
    """
    Execute predefined analytics queries and export results.
    
    Designed for:
    - Automated dashboard generation
    - Monthly reporting
    - Ad-hoc analysis
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize with database manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self.queries = MerchantAnalyticsQueries()
    
    def run_query(self, query_name: str) -> pd.DataFrame:
        """
        Run a predefined analytics query.
        
        Args:
            query_name: Name of the query from MerchantAnalyticsQueries
            
        Returns:
            DataFrame with query results
        """
        all_queries = self.queries.get_all_queries()
        
        if query_name not in all_queries:
            raise ValueError(f"Unknown query: {query_name}. Available: {list(all_queries.keys())}")
        
        query_result = all_queries[query_name]
        logger.info(f"Executing query: {query_name}")
        logger.info(f"Description: {query_result.description}")
        
        return self.db.execute_query(query_result.sql)
    
    def run_all_queries(self) -> Dict[str, pd.DataFrame]:
        """
        Run all predefined analytics queries.
        
        Returns:
            Dictionary of query names to result DataFrames
        """
        results = {}
        all_queries = self.queries.get_all_queries()
        
        for name in all_queries.keys():
            try:
                results[name] = self.run_query(name)
                logger.info(f"Query {name}: {len(results[name])} rows")
            except Exception as e:
                logger.error(f"Error in query {name}: {e}")
                results[name] = pd.DataFrame()
        
        return results
    
    def export_for_tableau(
        self, 
        output_dir: str = 'dashboards/tableau_exports',
        queries: Optional[List[str]] = None
    ) -> List[str]:
        """
        Export query results to CSV for Tableau import.
        
        Args:
            output_dir: Directory for output files
            queries: List of query names (None = all queries)
            
        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_queries = self.queries.get_all_queries()
        target_queries = queries or list(all_queries.keys())
        
        exported_files = []
        
        for name in target_queries:
            try:
                df = self.run_query(name)
                
                # Export to CSV
                csv_path = output_path / f"{name}.csv"
                df.to_csv(csv_path, index=False)
                exported_files.append(str(csv_path))
                
                logger.info(f"Exported {name} to {csv_path}")
                
            except Exception as e:
                logger.error(f"Failed to export {name}: {e}")
        
        # Create manifest file for Tableau
        manifest = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'files': exported_files,
            'query_descriptions': {
                name: all_queries[name].description 
                for name in target_queries
            }
        }
        
        manifest_path = output_path / 'manifest.json'
        pd.DataFrame([manifest]).to_json(manifest_path, orient='records', indent=2)
        
        return exported_files
    
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate executive summary combining key metrics.
        
        Returns:
            DataFrame with key business metrics
        """
        summary_query = """
        SELECT 
            (SELECT COUNT(*) FROM merchants) as total_merchants,
            (SELECT COUNT(*) FROM users) as total_users,
            (SELECT COUNT(*) FROM transactions) as total_transactions,
            (SELECT ROUND(SUM(amount_usd), 2) FROM transactions WHERE status = 'completed') as total_volume_usd,
            (SELECT ROUND(SUM(total_fee_usd), 2) FROM transactions WHERE status = 'completed') as total_fees_usd,
            (SELECT ROUND(AVG(amount_usd), 2) FROM transactions) as avg_transaction_size,
            (SELECT COUNT(*) FROM merchants WHERE churn_risk_score > 0.2) as high_risk_merchants,
            (SELECT ROUND(SUM(amount_usd), 2) FROM transactions WHERE is_cross_border = 1) as cross_border_volume
        """
        
        return self.db.execute_query(summary_query)


def setup_database(data_dir: str = 'data/raw', db_path: str = 'data/analytics.db') -> DatabaseManager:
    """
    Set up database with sample data.
    
    Args:
        data_dir: Directory containing CSV files
        db_path: Path for SQLite database
        
    Returns:
        Configured DatabaseManager instance
    """
    db = DatabaseManager(db_path)
    
    # Load all CSV data
    loaded = db.load_all_data(data_dir)
    print(f"Loaded tables: {loaded}")
    
    # Create indexes
    db.create_indexes()
    
    return db


def main():
    """Demo: Set up database and run sample queries."""
    # Initialize database
    db = setup_database()
    
    # Create query executor
    executor = AnalyticsQueryExecutor(db)
    
    # Run all queries and display summaries
    print("\n" + "=" * 60)
    print("RUNNING ANALYTICS QUERIES")
    print("=" * 60)
    
    results = executor.run_all_queries()
    
    for name, df in results.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        if len(df) > 0:
            print(df.head(5).to_string())
        else:
            print("No results")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    summary = executor.generate_summary_report()
    print(summary.T.to_string())
    
    # Export for Tableau
    exported = executor.export_for_tableau()
    print(f"\nExported {len(exported)} files for Tableau")


if __name__ == "__main__":
    main()
