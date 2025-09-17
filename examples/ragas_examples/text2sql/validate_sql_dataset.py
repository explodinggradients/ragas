#!/usr/bin/env python3
"""
SQL Dataset Validation Script

This script validates the Text-to-SQL dataset by executing each SQL query
against the database and capturing results for manual verification.

Usage:
    python validate_sql_dataset.py
    
Output:
    - validation_results.json: Detailed results for each query
    - validation_summary.json: Summary statistics
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Import our database utilities
from .db_utils import SQLiteDB, execute_sql


def load_dataset(csv_path: str = "datasets/booksql_sample.csv") -> List[Dict[str, Any]]:
    """
    Load the SQL dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing queries
        
    Returns:
        List of dictionaries containing query data
    """
    dataset = []
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dataset.append({
                'index': i,
                'query': row['Query'].strip(),
                'sql': row['SQL'].strip(),
                'level': row['Levels'].strip(),
                'split': row['split'].strip()
            })
    
    return dataset


def execute_and_validate_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single SQL query and capture results.
    
    Args:
        query_data: Dictionary containing query information
        
    Returns:
        Dictionary with execution results
    """
    result = {
        'index': query_data['index'],
        'natural_language_query': query_data['query'],
        'sql_query': query_data['sql'],
        'difficulty_level': query_data['level'],
        'dataset_split': query_data['split'],
        'execution_success': False,
        'execution_time': None,
        'error_message': None,
        'result_data': None,
        'result_shape': None,
        'result_columns': None
    }
    
    # Record execution time
    start_time = datetime.now()
    
    try:
        # Execute the SQL query with case-insensitive string matching
        success, query_result = execute_sql(query_data['sql'], case_insensitive=True)
        
        end_time = datetime.now()
        result['execution_time'] = (end_time - start_time).total_seconds()
        
        if success and isinstance(query_result, pd.DataFrame):
            result['execution_success'] = True
            result['result_shape'] = list(query_result.shape)  # [rows, columns]
            result['result_columns'] = list(query_result.columns)
            
            # Convert DataFrame to list of dictionaries for JSON serialization
            # Limit to first 10 rows to keep output manageable
            if len(query_result) > 10:
                sample_data = query_result.head(10)
                result['result_data'] = sample_data.to_dict('records')
                result['result_truncated'] = True
                result['total_rows'] = len(query_result)
            else:
                result['result_data'] = query_result.to_dict('records')
                result['result_truncated'] = False
                result['total_rows'] = len(query_result)
            
            # Classify result type for better reporting
            if len(query_result) == 0:
                result['result_type'] = 'empty'
            elif len(query_result) > 0:
                first_row = query_result.iloc[0]
                # Check if all values in the first row are null/None
                if all(pd.isna(value) or value is None for value in first_row):
                    result['result_type'] = 'null_values'
                else:
                    result['result_type'] = 'has_data'
            else:
                result['result_type'] = 'has_data'
        else:
            result['execution_success'] = False
            result['error_message'] = str(query_result)
            result['result_type'] = 'failed'
            
    except Exception as e:
        end_time = datetime.now()
        result['execution_time'] = (end_time - start_time).total_seconds()
        result['execution_success'] = False
        result['error_message'] = f"Unexpected error: {str(e)}"
        result['result_type'] = 'failed'
    
    return result


def generate_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from validation results.
    
    Args:
        results: List of validation results
        
    Returns:
        Dictionary containing summary statistics
    """
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r['execution_success'])
    failed_queries = total_queries - successful_queries
    
    # Count by result type
    result_type_counts = {
        'has_data': sum(1 for r in results if r.get('result_type') == 'has_data'),
        'null_values': sum(1 for r in results if r.get('result_type') == 'null_values'),
        'empty': sum(1 for r in results if r.get('result_type') == 'empty'),
        'failed': sum(1 for r in results if r.get('result_type') == 'failed')
    }
    
    # Group by difficulty level
    level_stats = {}
    for result in results:
        level = result['difficulty_level']
        if level not in level_stats:
            level_stats[level] = {
                'total': 0, 'successful': 0, 'failed': 0,
                'has_data': 0, 'null_values': 0, 'empty': 0
            }
        
        level_stats[level]['total'] += 1
        if result['execution_success']:
            level_stats[level]['successful'] += 1
        else:
            level_stats[level]['failed'] += 1
        
        # Count by result type for this level
        result_type = result.get('result_type', 'unknown')
        if result_type in level_stats[level]:
            level_stats[level][result_type] += 1
    
    # Calculate success rates
    for level in level_stats:
        total = level_stats[level]['total']
        successful = level_stats[level]['successful']
        level_stats[level]['success_rate'] = successful / total if total > 0 else 0
    
    # Common error types
    error_types = {}
    for result in results:
        if not result['execution_success'] and result['error_message']:
            # Extract first part of error message as error type
            error_type = result['error_message'].split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # Average execution time
    execution_times = [r['execution_time'] for r in results if r['execution_time'] is not None]
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    summary = {
        'validation_timestamp': datetime.now().isoformat(),
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'overall_success_rate': successful_queries / total_queries if total_queries > 0 else 0,
        'average_execution_time_seconds': avg_execution_time,
        'result_type_counts': result_type_counts,
        'statistics_by_difficulty': level_stats,
        'common_error_types': error_types,
        'sample_successful_queries': [
            r['index'] for r in results if r['execution_success']
        ][:5],  # First 5 successful queries
        'sample_failed_queries': [
            r['index'] for r in results if not r['execution_success']
        ][:5]   # First 5 failed queries
    }
    
    return summary


def main():
    """Main validation script."""
    print("🔍 Starting SQL Dataset Validation...")
    print("=" * 50)
    
    # Load dataset
    try:
        dataset = load_dataset("datasets/booksql_sample.csv")
        print(f"📊 Loaded {len(dataset)} queries from dataset")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading dataset: {e}")
        return
    
    # Validate database connection
    print("🔗 Testing database connection...")
    db = SQLiteDB()
    success, message = db.connect()
    if not success:
        print(f"❌ Database connection failed: {message}")
        print("💡 Make sure the BookSQL database is available at: BookSQL-files/BookSQL/accounting.sqlite")
        return
    
    # Get database info
    success, tables = db.get_table_names()
    if success:
        print(f"✅ Database connected. Found tables: {tables}")
    db.disconnect()
    
    # Execute all queries
    print(f"\n🚀 Executing {len(dataset)} SQL queries...")
    results = []
    
    for i, query_data in enumerate(dataset):
        print(f"Processing query {i+1}/{len(dataset)}: {query_data['level']} level", end=" ... ")
        
        result = execute_and_validate_query(query_data)
        results.append(result)
        
        if result['execution_success']:
            print("✅")
        else:
            print("❌")
    
    # Generate summary
    print("\n📈 Generating summary statistics...")
    summary = generate_summary_statistics(results)
    
    # Save results
    print("💾 Saving validation results...")
    
    # Save detailed results
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    with open('validation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful_queries']} ({summary['overall_success_rate']:.1%})")
    print(f"Failed: {summary['failed_queries']}")
    print(f"Average Execution Time: {summary['average_execution_time_seconds']:.3f}s")
    
    print("\n📈 Result Type Distribution:")
    result_counts = summary['result_type_counts']
    total = summary['total_queries']
    print(f"  ✅ Has Data: {result_counts['has_data']}/{total} ({result_counts['has_data']/total:.1%})")
    print(f"  🔍 NULL Values: {result_counts['null_values']}/{total} ({result_counts['null_values']/total:.1%})")
    print(f"  📭 Empty Results: {result_counts['empty']}/{total} ({result_counts['empty']/total:.1%})")
    print(f"  ❌ Failed: {result_counts['failed']}/{total} ({result_counts['failed']/total:.1%})")
    
    print("\n📈 Success Rate by Difficulty:")
    for level, stats in summary['statistics_by_difficulty'].items():
        print(f"  {level.capitalize()}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")
        print(f"    ✅ Data: {stats['has_data']}, 🔍 NULL: {stats['null_values']}, 📭 Empty: {stats['empty']}, ❌ Failed: {stats['failed']}")
    
    if summary['common_error_types']:
        print("\n⚠️  Common Error Types:")
        for error_type, count in sorted(summary['common_error_types'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error_type}: {count} occurrences")
    
    print("\n💾 Detailed results saved to:")
    print("  - validation_results.json (detailed results)")
    print("  - validation_summary.json (summary statistics)")
    
    if summary['failed_queries'] > 0:
        print("\n🔍 Review failed queries in validation_results.json")
        print("💡 Check if database schema matches expected tables/columns")


if __name__ == "__main__":
    main()
