#!/usr/bin/env python3
"""
Database utilities for Text-to-SQL evaluation with Ragas.

This module provides a simple SQLite database interface for executing
SQL queries and returning results in pandas DataFrame format, optimized
for use with datacompy for result comparison.
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")


class SQLiteDB:
    """
    Simple SQLite database interface for text-to-SQL evaluation.
    
    Designed to work seamlessly with datacompy for DataFrame comparison.
    Handles connection management and provides clear error reporting for agents.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite database connection.
        
        Args:
            db_path: Path to SQLite database file. 
                    Defaults to BookSQL dataset: "BookSQL-files/BookSQL/accounting.sqlite"
        """
        if db_path is None:
            # Default to BookSQL dataset path
            self.db_path = Path("BookSQL-files/BookSQL/accounting.sqlite")
        else:
            self.db_path = Path(db_path)
            
        self._connection = None
        
    def connect(self) -> Tuple[bool, str]:
        """
        Establish database connection.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.db_path.exists():
                return False, f"Database file not found: {self.db_path}"
                
            # Use a small connection timeout to avoid indefinite waits on locked DB
            self._connection = sqlite3.connect(str(self.db_path), timeout=1.0)
            # Enable row factory for column names
            self._connection.row_factory = sqlite3.Row
            # Ensure we also have a short busy timeout inside SQLite
            try:
                self._connection.execute("PRAGMA busy_timeout = 1000")
            except Exception:
                # Best-effort; ignore if not supported
                pass
            return True, "Connected successfully"
            
        except sqlite3.Error as e:
            return False, f"Database connection error: {e}"
        except Exception as e:
            return False, f"Unexpected error connecting to database: {e}"
    
    def disconnect(self) -> None:
        """Close database connection if open."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def execute_query(self, sql: str, replace_current_date: bool = True, case_insensitive: bool = True) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a SELECT query and return results as pandas DataFrame.
        
        Args:
            sql: SQL SELECT query string
            replace_current_date: Whether to replace current_date/now with fixed date (default: True)
            case_insensitive: Whether to make string case-insensitive (default: False)
            
        Returns:
            Tuple of (success: bool, result: DataFrame | error_message: str)
            
        Note:
            - On success: (True, pandas.DataFrame)
            - On failure: (False, error_message_string)
            - current_date is replaced with '2022-06-01' to work with historical data
            - case_insensitive=True converts string literals and adds LOWER() comparisons
        """
        # Ensure we have a connection
        if not self._connection:
            success, message = self.connect()
            if not success:
                return False, f"Connection failed: {message}"
        
        # Basic SQL injection prevention - only allow SELECT queries
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith('SELECT'):
            return False, "Only SELECT queries are supported"
        
        # Apply comprehensive SQL normalization pipeline
        
        # 1. Normalize quotes for database execution  
        sql = self._normalize_quotes_for_execution(sql)
        
        # 2. Normalize whitespace
        sql = self._normalize_whitespace(sql)
        
        # 3. Replace current_date and now with fixed date for historical data compatibility
        if replace_current_date:
            sql = self._replace_current_date_and_now(sql, '2022-06-01')
        
        # 4. Make string comparisons case-insensitive if requested
        if case_insensitive:
            sql = sql.lower()
            
        try:
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(sql, self._connection)
            return True, df
            
        except sqlite3.OperationalError as e:
            # SQL syntax or runtime errors
            return False, f"SQL execution error: {e}"
        except sqlite3.DatabaseError as e:
            # Database-level errors
            return False, f"Database error: {e}"
        except pd.errors.DatabaseError as e:
            # Pandas-specific database errors
            return False, f"Pandas database error: {e}"
        except Exception as e:
            # Catch-all for unexpected errors
            return False, f"Unexpected error executing query: {e}"
    
    def _replace_current_date_and_now(self, sql: str, fixed_date: str) -> str:
        """
        Replace current_date and now() with a fixed date for historical data compatibility.
        
        Args:
            sql: Original SQL query
            fixed_date: Fixed date string (e.g., '2022-06-01')
            
        Returns:
            Modified SQL query with replaced date functions
        """
        # Replace current_date with quoted fixed date
        sql = sql.replace('current_date', f"'{fixed_date}'")
        
        # Replace now() and 'now' with quoted fixed date  
        sql = sql.replace(', now', f", '{fixed_date}'")
        sql = sql.replace("'now'", f"'{fixed_date}'")
        
        # Also handle %Y date formatting issues (from original code)
        sql = sql.replace('%y', "%Y")
        
        return sql
    
    def _normalize_quotes_for_execution(self, sql: str) -> str:
        """
        Standardize quotes for database execution: double quotes → single quotes.
        
        Args:
            sql: Original SQL query
            
        Returns:
            SQL with standardized single quotes
        """
        return sql.replace('"', "'")
    
    def _normalize_whitespace(self, sql: str) -> str:
        """
        Normalize whitespace in SQL query.
        
        Args:
            sql: Original SQL query
            
        Returns:
            SQL with normalized whitespace (multiple spaces → single space)
        """
        return re.sub(r'\s+', ' ', sql.strip())
    
    def get_schema_info(self) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Get database schema information as DDL statements.
        
        This provides the most comprehensive and agent-friendly schema information,
        showing the actual CREATE TABLE/VIEW statements that define the database structure.
        
        Returns:
            Tuple of (success: bool, result: DataFrame | error_message: str)
            DataFrame contains: name, type, sql (DDL statements)
        """
        schema_query = """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
        return self.execute_query(schema_query, replace_current_date=False, case_insensitive=False)
    
    def get_table_names(self) -> Tuple[bool, Union[list, str]]:
        """
        Get list of all table names in the database.
        
        Returns:
            Tuple of (success: bool, result: list | error_message: str)
        """
        tables_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        success, result = self.execute_query(tables_query)
        
        if success and isinstance(result, pd.DataFrame):
            table_names = result['name'].tolist()
            return True, table_names
        else:
            return False, str(result)
    
    def normalize_sql(self, sql: str, replace_current_date: bool = True, case_insensitive: bool = True) -> str:
        """
        Apply the same normalization pipeline used during query execution.
        
        Useful for comparing SQL queries or preparing them for external analysis.
        
        Args:
            sql: SQL query string to normalize
            replace_current_date: Whether to replace current_date/now with fixed date
            case_insensitive: Whether to make case-insensitive
            
        Returns:
            Normalized SQL query string
        """
        # Apply same normalization pipeline as execute_query
        
        # 1. Normalize quotes for consistency
        sql = self._normalize_quotes_for_execution(sql)
        
        # 2. Normalize whitespace
        sql = self._normalize_whitespace(sql)
        
        # 3. Replace date functions if requested
        if replace_current_date:
            sql = self._replace_current_date_and_now(sql, '2022-06-01')
        
        # 4. Apply case insensitivity if requested
        if case_insensitive:
            sql = sql.lower()
            
        return sql
    
    def __enter__(self):
        """Context manager entry."""
        success, message = self.connect()
        if not success:
            raise RuntimeError(f"Failed to connect to database: {message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience functions for quick usage
def execute_sql(sql: str, db_path: Optional[str] = None, replace_current_date: bool = True, case_insensitive: bool = True) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Execute a single SQL query with automatic connection management.
    
    Args:
        sql: SQL SELECT query string
        db_path: Optional database path (uses BookSQL default if None)
        replace_current_date: Whether to replace current_date/now with fixed date (default: True)
        case_insensitive: Whether to make string comparisons case-insensitive (default: False)
        
    Returns:
        Tuple of (success: bool, result: DataFrame | error_message: str)
    """
    # Ensure connection is always closed after the query
    with SQLiteDB(db_path) as db:
        return db.execute_query(sql, replace_current_date, case_insensitive)


def get_database_schema(db_path: Optional[str] = None) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Get database schema information as DDL statements with automatic connection management.
    
    Args:
        db_path: Optional database path (uses BookSQL default if None)
        
    Returns:
        Tuple of (success: bool, result: DataFrame | error_message: str)
        DataFrame contains: name, type, sql (DDL statements)
    """
    with SQLiteDB(db_path) as db:
        return db.get_schema_info()


def normalize_sql_query(sql: str, replace_current_date: bool = True, case_insensitive: bool = True) -> str:
    """
    Normalize a SQL query using the same pipeline as query execution.
    
    Args:
        sql: SQL query string to normalize
        replace_current_date: Whether to replace current_date/now with fixed date (default: True)
        case_insensitive: Whether to make case-insensitive (default: True)
        
    Returns:
        Normalized SQL query string
        
    Example:
        >>> sql = 'SELECT   *  FROM "customers"   WHERE   name   =   "John"'
        >>> normalize_sql_query(sql)
        "select * from 'customers' where name = 'john'"
    """
    db = SQLiteDB()
    return db.normalize_sql(sql, replace_current_date, case_insensitive)


def main():
    """Main CLI interface for executing SQL queries."""
    parser = argparse.ArgumentParser(
        description="Execute SQL queries against SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute a simple query
  python db_utils.py --query "SELECT COUNT(*) FROM master_txn_table"
  
  # Execute a complex join query
  python db_utils.py --query "SELECT sum(debit) FROM master_txn_table AS T1 JOIN chart_of_accounts AS T2 ON T1.account = T2.account_name WHERE account_type IN ('Expense', 'Other Expense')"
  
  # Use custom database path
  python db_utils.py --db /path/to/database.sqlite --query "SELECT * FROM users LIMIT 5"
  
  # Get schema information
  python db_utils.py --schema
  
  # List all tables
  python db_utils.py --tables
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="SQL query to execute (SELECT queries only)"
    )
    
    parser.add_argument(
        "--db", "-d",
        type=str,
        help="Path to SQLite database file (defaults to BookSQL dataset)"
    )
    
    parser.add_argument(
        "--schema", "-s",
        action="store_true",
        help="Show database schema information"
    )
    
    parser.add_argument(
        "--tables", "-t",
        action="store_true",
        help="List all tables in the database"
    )
    
    parser.add_argument(
        "--no-date-replacement",
        action="store_true",
        help="Disable automatic replacement of current_date with fixed date"
    )
    
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Make string comparisons case-insensitive"
    )
    
    args = parser.parse_args()
    
    # Check if any action was specified
    if not any([args.query, args.schema, args.tables]):
        parser.print_help()
        print("\nError: You must specify at least one action (--query, --schema, or --tables)")
        sys.exit(1)
    
    try:
        with SQLiteDB(args.db) as db:
            
            # Handle schema request
            if args.schema:
                print("=== Database Schema ===")
                success, result = db.get_schema_info()
                if success and isinstance(result, pd.DataFrame):
                    print(result.to_string(index=False))
                else:
                    print(f"Error getting schema: {result}")
                    sys.exit(1)
                print()  # Add spacing
            
            # Handle tables request
            if args.tables:
                print("=== Database Tables ===")
                success, tables = db.get_table_names()
                if success:
                    for table in tables:
                        print(f"  {table}")
                else:
                    print(f"Error getting tables: {tables}")
                    sys.exit(1)
                print()  # Add spacing
            
            # Handle query execution
            if args.query:
                print("=== Query Results ===")
                print(f"Query: {args.query}")
                print()
                
                success, result = db.execute_query(
                    args.query,
                    replace_current_date=not args.no_date_replacement,
                    case_insensitive=args.case_insensitive
                )
                
                if success and isinstance(result, pd.DataFrame):
                    if len(result) == 0:
                        print("Query executed successfully but returned no rows.")
                    else:
                        print(result.to_string(index=False))
                        print(f"\nRows returned: {len(result)}")
                else:
                    print(f"Query failed: {result}")
                    sys.exit(1)
                    
    except RuntimeError as e:
        print(f"Database connection error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
