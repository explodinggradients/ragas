#!/usr/bin/env python3
"""
Data utilities for Text-to-SQL evaluation with Ragas.

This module provides CLI tools to download and prepare datasets for 
text-to-SQL evaluation workflows.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)

try:
    import pandas as pd
    from pandas import DataFrame
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

# Import validation functions from validate_sql_dataset.py
try:
    from .validate_sql_dataset import execute_and_validate_query
except ImportError:
    print("Error: validate_sql_dataset.py not found in the same directory")
    sys.exit(1)


def download_booksql_dataset() -> bool:
    """
    Download the BookSQL dataset from Hugging Face Hub to ./BookSQL-files directory.
        
    Returns:
        bool: True if download successful, False otherwise
        
    Note:
        This dataset is gated and requires accepting terms on the Hugging Face Hub.
        You need to:
        1. Visit https://huggingface.co/datasets/Exploration-Lab/BookSQL
        2. Accept the terms and conditions
        3. Authenticate with: huggingface-cli login
    """
    repo_id = "Exploration-Lab/BookSQL"
    local_dir = "BookSQL-files"
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading BookSQL dataset to {local_dir}...")
    print(f"📂 Repository: {repo_id}")
    
    try:
        # Download the entire repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Create actual files, not symlinks
        )
        
        print(f"✅ Successfully downloaded dataset to: {downloaded_path}")
        
        # List downloaded files
        dataset_path = Path(local_dir)
        files = list(dataset_path.rglob("*"))
        print(f"📁 Downloaded {len(files)} files:")
        for file in sorted(files)[:10]:  # Show first 10 files
            if file.is_file():
                print(f"   • {file.relative_to(dataset_path)}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
            
        return True
        
    except GatedRepoError:
        print("❌ Error: This dataset is gated and requires authentication.")
        print("Please follow these steps:")
        print("1. Visit: https://huggingface.co/datasets/Exploration-Lab/BookSQL")
        print("2. Accept the terms and conditions")
        print("3. Run: huggingface-cli login")
        print("4. Try downloading again")
        return False
        
    except RepositoryNotFoundError:
        print(f"❌ Error: Repository '{repo_id}' not found.")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False


def validate_query_data(query_data: Dict[str, Any], require_data: bool = False) -> bool:
    """
    Validate a single query by executing it against the database.
    
    Args:
        query_data: Dictionary containing query information (query, sql, level, split)
        require_data: If True, only accept queries that return actual data
        
    Returns:
        bool: True if query is valid (and optionally returns data), False otherwise
    """
    try:
        result = execute_and_validate_query(query_data)
        
        if not result['execution_success']:
            return False
            
        if require_data:
            # Only accept queries that return actual data (not empty or null values)
            return result.get('result_type') == 'has_data'
        else:
            # Accept any successful query execution
            return True
            
    except Exception as e:
        print(f"   ⚠️  Error validating query: {e}")
        return False


def create_sample_dataset(
    input_file: str = "BookSQL-files/BookSQL/train.json",
    output_dir: str = "datasets",
    output_filename: str = "booksql_sample.csv",
    samples_per_level: int = 10,
    random_seed: int = 42,
    validate_queries: bool = False,
    require_data: bool = False
) -> bool:
    """
    Create a balanced sample dataset from BookSQL train.json.
    
    Args:
        input_file: Path to the BookSQL train.json file
        output_dir: Directory to save the output CSV
        output_filename: Name of the output CSV file
        samples_per_level: Number of samples per difficulty level (easy, medium, hard)
        random_seed: Random seed for reproducible sampling
        validate_queries: If True, validate SQL queries before including them
        require_data: If True (and validate_queries=True), only include queries that return data
        
    Returns:
        bool: True if successful, False otherwise
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_file}' not found.")
        print("💡 Tip: Run with --download-data first to download the BookSQL dataset.")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Loading data from {input_file}...")
    
    try:
        # Load JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data)} total records")
        
        # Convert to DataFrame
        df: DataFrame = pd.DataFrame(data)
        
        # Filter for train split only
        train_df = df[df['split'] == 'train'].copy()  # DataFrame with train split only
        print(f"🚂 Found {len(train_df)} train records")
        
        # Remove duplicates based on Query + SQL combination
        # Keep track of original count for reporting
        original_count = len(train_df)
        train_df = train_df.drop_duplicates(subset=['Query', 'SQL'], keep='first')  # type: ignore
        duplicate_count = original_count - len(train_df)
        
        if duplicate_count > 0:
            print(f"🔍 Removed {duplicate_count} duplicate records (same Query + SQL)")
            print(f"📊 {len(train_df)} unique records remaining")
        
        # Check available difficulty levels
        level_counts = train_df['Levels'].value_counts()  # type: ignore
        print("📈 Difficulty distribution (after deduplication):")
        for level, count in level_counts.items():
            print(f"   • {level}: {count} records")
        
        # Sample data for each difficulty level
        sampled_dfs = []
        
        if validate_queries:
            print("🔍 Validation enabled - testing SQL queries before including them in sample")
            if require_data:
                print("📊 Only including queries that return actual data")
        
        for level in ['easy', 'medium', 'hard']:
            level_data = train_df[train_df['Levels'] == level]
            
            if len(level_data) == 0:
                print(f"⚠️  Warning: No '{level}' records found, skipping...")
                continue
            
            if not validate_queries:
                # Original sampling logic without validation
                if len(level_data) < samples_per_level:
                    print(f"⚠️  Warning: Only {len(level_data)} '{level}' records available, using all of them")
                    sampled = level_data
                else:
                    # Random sampling with fixed seed - explicit DataFrame operation
                    sampled = level_data.sample(  # type: ignore
                        n=samples_per_level, 
                        random_state=random_seed
                    )
                    
                sampled_dfs.append(sampled)
                print(f"✅ Sampled {len(sampled)} '{level}' records")
            else:
                # Validation-based sampling
                print(f"🔍 Validating '{level}' queries to find {samples_per_level} valid samples...")
                
                # Shuffle the level data to get random samples during validation
                shuffled_level_data = level_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                
                valid_samples = []
                checked_count = 0
                
                for idx, row in shuffled_level_data.iterrows():
                    checked_count += 1
                    print(f"   Testing {level} query {checked_count}/{len(shuffled_level_data)}...", end="")
                    
                    # Prepare query data for validation
                    query_data = {
                        'index': idx,
                        'query': row['Query'],
                        'sql': row['SQL'],
                        'level': row['Levels'],
                        'split': row['split']
                    }
                    
                    if validate_query_data(query_data, require_data):
                        valid_samples.append(row)
                        print(" ✅")
                        
                        # Stop if we have enough samples
                        if len(valid_samples) >= samples_per_level:
                            break
                    else:
                        print(" ❌")
                
                if len(valid_samples) == 0:
                    print(f"⚠️  Warning: No valid '{level}' queries found, skipping this level...")
                    continue
                elif len(valid_samples) < samples_per_level:
                    print(f"⚠️  Warning: Only found {len(valid_samples)} valid '{level}' queries out of {samples_per_level} requested")
                else:
                    print(f"✅ Found {len(valid_samples)} valid '{level}' queries")
                
                # Convert valid samples back to DataFrame
                if valid_samples:
                    sampled = pd.DataFrame(valid_samples)
                    sampled_dfs.append(sampled)
                    print(f"✅ Added {len(sampled)} validated '{level}' records")
        
        if not sampled_dfs:
            print("❌ Error: No data could be sampled")
            return False
        
        # Combine all sampled data
        final_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Final duplicate check (safety measure)
        pre_final_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['Query', 'SQL'], keep='first')  # type: ignore
        final_duplicate_count = pre_final_count - len(final_df)
        
        if final_duplicate_count > 0:
            print(f"⚠️  Warning: Removed {final_duplicate_count} duplicates from final sample")
        
        # Shuffle the final dataset to mix difficulty levels
        final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Save to CSV
        output_file_path = output_path / output_filename
        final_df.to_csv(output_file_path, index=False)
        
        print(f"💾 Saved {len(final_df)} records to {output_file_path}")
        print("📋 Final distribution:")
        for level, count in final_df['Levels'].value_counts().items():
            print(f"   • {level}: {count} records")
            
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {input_file}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data utilities for Text-to-SQL evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --download-data                                    # Download BookSQL dataset to ./BookSQL-files
  %(prog)s --create-sample                                    # Create sample CSV with 15 easy/medium/hard cases
  %(prog)s --create-sample --samples 5                       # Create sample CSV with 5 easy/medium/hard cases
  %(prog)s --create-sample --validate                        # Create sample with SQL validation (any executable query)
  %(prog)s --create-sample --validate --require-data         # Create sample with only queries that return data
  %(prog)s --create-sample --samples 10 --validate --require-data  # 10 validated queries per level that return data
        """
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the BookSQL dataset to ./BookSQL-files directory"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a balanced sample CSV from BookSQL train.json (15 each: easy, medium, hard)"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of samples per difficulty level (default: 15)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate SQL queries before including them in the sample (requires database connection)"
    )
    
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Only include queries that return actual data (requires --validate). Excludes empty results and null values."
    )
    
    args = parser.parse_args()
    
    if args.download_data:
        success = download_booksql_dataset()
        sys.exit(0 if success else 1)
    elif args.create_sample:
        # Validate argument combinations
        if args.require_data and not args.validate:
            print("❌ Error: --require-data requires --validate to be enabled")
            sys.exit(1)
            
        success = create_sample_dataset(
            samples_per_level=args.samples,
            validate_queries=args.validate,
            require_data=args.require_data
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
