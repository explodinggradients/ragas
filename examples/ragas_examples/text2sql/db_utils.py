#!/usr/bin/env python3
"""
Simple database utilities for Text-to-SQL evaluation.

This module helps you execute SQL queries against SQLite databases 
and get results as pandas DataFrames for easy comparison in evaluations.
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
    
    This class makes it easy to:
    - Connect to SQLite databases  
    - Execute SQL queries
    - Get results as pandas DataFrames
    - Handle errors gracefully
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Create a new database connection.
        
        Args:
            db_path: Path to SQLite database file.
                    If None, uses BookSQL dataset: "BookSQL-files/BookSQL/accounting.sqlite"
        """
        if db_path is None:
            self.db_path = Path("BookSQL-files/BookSQL/accounting.sqlite")
        else:
            self.db_path = Path(db_path)
            
        self._connection = None
        
    def connect(self) -> Tuple[bool, str]:
        """
        Connect to the database.
        
        Returns:
            (success: bool, message: str)
        """
        try:
            if not self.db_path.exists():
                return False, f"Database file not found: {self.db_path}"
                
            self._connection = sqlite3.connect(str(self.db_path), timeout=1.0)
            self._connection.row_factory = sqlite3.Row
            return True, "Connected successfully"
            
        except Exception as e:
            return False, f"Database connection error: {e}"
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def execute_query(self, sql: str, replace_current_date: bool = True, case_insensitive: bool = True) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a SQL query and return results as a DataFrame.
        
        Args:
            sql: SQL SELECT query to execute
            replace_current_date: Replace date functions with fixed date for historical data
            case_insensitive: Make string comparisons case-insensitive
            
        Returns:
            (success: bool, result: DataFrame or error_message: str)
            
        Example:
            success, result = db.execute_query("SELECT COUNT(*) FROM customers")
            if success:
                print(f"Found {result.iloc[0, 0]} customers")
            else:
                print(f"Query failed: {result}")
        """
        # Connect if needed
        if not self._connection:
            success, message = self.connect()
            if not success:
                return False, f"Connection failed: {message}"
        
        # Security check - only allow SELECT queries
        if not sql.strip().upper().startswith('SELECT'):
            return False, "Only SELECT queries are supported"
        
        # Clean up the SQL query
        sql = self._normalize_sql(sql, replace_current_date, case_insensitive)
            
        try:
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(sql, self._connection)
            return True, df
            
        except Exception as e:
            return False, f"SQL execution error: {e}"
    
    def _normalize_sql(self, sql: str, replace_current_date: bool, case_insensitive: bool) -> str:
        """
        Clean up SQL query for better compatibility.
        
        This method:
        - Fixes quote marks (double → single)
        - Cleans up whitespace
        - Replaces date functions with fixed dates 
        - Makes text case-insensitive if requested
        """
        # Fix quotes: double → single
        sql = sql.replace('"', "'")
        
        # Clean up whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Replace date functions with fixed date for historical data
        if replace_current_date:
            sql = sql.replace('current_date', "'2022-06-01'")
            sql = sql.replace(', now', ", '2022-06-01'")
            sql = sql.replace("'now'", "'2022-06-01'")
            sql = sql.replace('%y', "%Y")
        
        # Make case-insensitive if requested
        if case_insensitive:
            sql = sql.lower()
            
        return sql
    
    def get_schema_info(self) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Get information about all tables and views in the database.
        
        Returns:
            (success: bool, schema_info: DataFrame or error_message: str)
            DataFrame contains: name, type, sql (CREATE statements)
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
        Get a list of all table names in the database.
        
        Returns:
            (success: bool, table_names: list or error_message: str)
        """
        tables_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        success, result = self.execute_query(tables_query, replace_current_date=False, case_insensitive=False)
        
        if success and isinstance(result, pd.DataFrame):
            return True, result['name'].tolist()
        else:
            return False, str(result)


# Convenience functions for quick usage

def execute_sql(sql: str, db_path: Optional[str] = None, replace_current_date: bool = True, case_insensitive: bool = True) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Execute a SQL query with automatic connection management.
    
    This is the main function you'll use for running SQL queries in evaluations.
    
    Args:
        sql: SQL SELECT query to execute
        db_path: Path to database file (uses BookSQL default if None)
        replace_current_date: Replace date functions with fixed date
        case_insensitive: Make string comparisons case-insensitive
        
    Returns:
        (success: bool, result: DataFrame or error_message: str)
        
    Example:
        success, data = execute_sql("SELECT COUNT(*) FROM customers")
        if success:
            print(f"Query returned {len(data)} rows")
        else:
            print(f"Error: {data}")
    """
    db = SQLiteDB(db_path)
    try:
        return db.execute_query(sql, replace_current_date, case_insensitive)
    finally:
        db.disconnect()


def get_database_schema(db_path: Optional[str] = None) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Get database schema information with automatic connection management.
    
    Args:
        db_path: Path to database file (uses BookSQL default if None)
        
    Returns:
        (success: bool, schema_info: DataFrame or error_message: str)
    """
    db = SQLiteDB(db_path)
    try:
        return db.get_schema_info()
    finally:
        db.disconnect()


def main():
    """Simple command-line interface for testing queries."""
    parser = argparse.ArgumentParser(
        description="Execute SQL queries against SQLite database",
        epilog="""
Examples:
  python db_utils.py --query "SELECT COUNT(*) FROM master_txn_table"
  python db_utils.py --schema
  python db_utils.py --tables
        """
    )
    
    parser.add_argument("--query", "-q", help="SQL query to execute")
    parser.add_argument("--db", "-d", help="Database file path")
    parser.add_argument("--schema", "-s", action="store_true", help="Show database schema")
    parser.add_argument("--tables", "-t", action="store_true", help="List all tables")
    
    args = parser.parse_args()
    
    # Must specify at least one action
    if not any([args.query, args.schema, args.tables]):
        parser.print_help()
        print("\nError: Specify --query, --schema, or --tables")
        sys.exit(1)
    
    try:
        db = SQLiteDB(args.db)
        
        # Show schema
        if args.schema:
            print("=== Database Schema ===")
            success, result = db.get_schema_info()
            if success:
                print(result.to_string(index=False))
            else:
                print(f"Error: {result}")
                sys.exit(1)
        
        # List tables
        if args.tables:
            print("=== Tables ===")
            success, tables = db.get_table_names()
            if success:
                for table in tables:
                    print(f"  {table}")
            else:
                print(f"Error: {tables}")
                sys.exit(1)
        
        # Execute query
        if args.query:
            print("=== Query Results ===")
            print(f"Query: {args.query}")
            print()
            
            success, result = db.execute_query(args.query)
            if success:
                if len(result) == 0:
                    print("No rows returned.")
                else:
                    print(result.to_string(index=False))
                    print(f"\nRows: {len(result)}")
            else:
                print(f"Error: {result}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.disconnect()


if __name__ == "__main__":
    main()