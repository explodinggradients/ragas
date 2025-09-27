#!/usr/bin/env python3
"""
Data utilities for Text-to-SQL evaluation with Ragas.

This module provides CLI tools to download and prepare datasets for 
text-to-SQL evaluation workflows.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Load environment variables from ragas root
try:
    from dotenv import load_dotenv
    # Load .env from ragas root directory (3 levels up from this file)
    ragas_root = Path(__file__).parent.parent.parent.parent
    env_path = ragas_root / ".env"
    load_dotenv(env_path)
except ImportError:
    # dotenv is optional, continue without it
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:
    logger.error("huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)

try:
    import pandas as pd
    from pandas import DataFrame
except ImportError:
    logger.error("pandas is required. Install with: pip install pandas")
    sys.exit(1)

# Import validation functions from validate_sql_dataset.py
try:
    from .validate_sql_dataset import execute_and_validate_query
except ImportError:
    logger.error("validate_sql_dataset.py not found in the same directory")
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
    
    logger.info(f"Downloading BookSQL dataset to {local_dir}")
    logger.info(f"Repository: {repo_id}")
    
    try:
        # Download the entire repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Create actual files, not symlinks
        )
        
        logger.info(f"Successfully downloaded dataset to: {downloaded_path}")
        
        # List downloaded files
        dataset_path = Path(local_dir)
        files = list(dataset_path.rglob("*"))
        logger.info(f"Downloaded {len(files)} files")
        for file in sorted(files)[:5]:  # Show first 5 files
            if file.is_file():
                logger.info(f"  {file.relative_to(dataset_path)}")
        if len(files) > 5:
            logger.info(f"  ... and {len(files) - 5} more files")
            
        return True
        
    except GatedRepoError:
        logger.error("This dataset is gated and requires authentication")
        logger.error("Please follow these steps:")
        logger.error("1. Visit: https://huggingface.co/datasets/Exploration-Lab/BookSQL")
        logger.error("2. Accept the terms and conditions")
        logger.error("3. Run: huggingface-cli login")
        logger.error("4. Try downloading again")
        return False
        
    except RepositoryNotFoundError:
        logger.error(f"Repository '{repo_id}' not found")
        return False
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
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
        logger.warning(f"Error validating query: {e}")
        return False


def load_and_clean_data(input_file: str) -> DataFrame:
    """
    Load JSON data and remove duplicates.
    
    Args:
        input_file: Path to the BookSQL train.json file
        
    Returns:
        DataFrame: Cleaned train data with duplicates removed
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    logger.info(f"Loading data from {input_file}")
    
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} total records")
    
    # Convert to DataFrame and filter for train split
    df = pd.DataFrame(data)
    train_df = df[df['split'] == 'train'].copy()
    logger.info(f"Found {len(train_df)} train records")
    
    # Remove duplicates based on Query + SQL combination
    original_count = len(train_df)
    train_df = train_df.drop_duplicates(subset=['Query', 'SQL'], keep='first')
    duplicate_count = original_count - len(train_df)
    
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate records")
    logger.info(f"{len(train_df)} unique records remaining")
    
    # Show difficulty distribution
    level_counts = train_df['Levels'].value_counts()
    logger.info("Difficulty distribution after deduplication:")
    for level, count in level_counts.items():
        logger.info(f"  {level}: {count} records")
    
    return train_df


def sample_by_difficulty(data: DataFrame, level: str, samples_per_level: int, random_seed: int) -> DataFrame:
    """
    Sample data for a specific difficulty level.
    
    Args:
        data: DataFrame containing the data
        level: Difficulty level ('easy', 'medium', 'hard')
        samples_per_level: Number of samples to take
        random_seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame: Sampled data for the specified level
    """
    level_data = data[data['Levels'] == level]
    
    if len(level_data) == 0:
        logger.warning(f"No '{level}' records found, skipping")
        return pd.DataFrame()
    
    if len(level_data) < samples_per_level:
        logger.warning(f"Only {len(level_data)} '{level}' records available, using all")
        return level_data
    else:
        sampled = level_data.sample(n=samples_per_level, random_state=random_seed)
        logger.info(f"Sampled {len(sampled)} '{level}' records")
        return sampled


def validate_samples(data: DataFrame, level: str, samples_per_level: int, 
                    random_seed: int, require_data: bool = False) -> DataFrame:
    """
    Sample and validate data for a specific difficulty level.
    
    Args:
        data: DataFrame containing the data
        level: Difficulty level ('easy', 'medium', 'hard')
        samples_per_level: Number of samples to find
        random_seed: Random seed for reproducible sampling
        require_data: If True, only include queries that return data
        
    Returns:
        DataFrame: Validated samples for the specified level
    """
    level_data = data[data['Levels'] == level]
    
    if len(level_data) == 0:
        logger.warning(f"No '{level}' records found, skipping")
        return pd.DataFrame()
    
    logger.info(f"Validating '{level}' queries to find {samples_per_level} valid samples")
    
    # Shuffle data for random sampling during validation
    shuffled_data = level_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    valid_samples = []
    checked_count = 0
    
    for idx, row in shuffled_data.iterrows():
        checked_count += 1
        
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
            
            # Stop if we have enough samples
            if len(valid_samples) >= samples_per_level:
                break
    
    if len(valid_samples) == 0:
        logger.warning(f"No valid '{level}' queries found, skipping this level")
        return pd.DataFrame()
    elif len(valid_samples) < samples_per_level:
        logger.warning(f"Only found {len(valid_samples)} valid '{level}' queries out of {samples_per_level} requested")
    else:
        logger.info(f"Found {len(valid_samples)} valid '{level}' queries")
    
    return pd.DataFrame(valid_samples) if valid_samples else pd.DataFrame()


def save_results(data: DataFrame, output_dir: str, output_filename: str, random_seed: int) -> bool:
    """
    Save final dataset to CSV.
    
    Args:
        data: Final dataset to save
        output_dir: Directory to save the output CSV
        output_filename: Name of the output CSV file
        random_seed: Random seed for final shuffle
        
    Returns:
        bool: True if successful, False otherwise
    """
    if data.empty:
        logger.error("No data to save")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Final duplicate check
    pre_final_count = len(data)
    data = data.drop_duplicates(subset=['Query', 'SQL'], keep='first')
    final_duplicate_count = pre_final_count - len(data)
    
    if final_duplicate_count > 0:
        logger.warning(f"Removed {final_duplicate_count} duplicates from final sample")
    
    # Shuffle the final dataset
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save to CSV
    output_file_path = output_path / output_filename
    data.to_csv(output_file_path, index=False)
    
    logger.info(f"Saved {len(data)} records to {output_file_path}")
    logger.info("Final distribution:")
    for level, count in data['Levels'].value_counts().items():
        logger.info(f"  {level}: {count} records")
    
    return True


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
    
    This function orchestrates the data loading, sampling, validation, and saving process.
    
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
    try:
        # Step 1: Load and clean data
        train_df = load_and_clean_data(input_file)
        
        # Step 2: Sample data for each difficulty level
        sampled_dfs = []
        
        if validate_queries:
            logger.info("Validation enabled - testing SQL queries before including them in sample")
            if require_data:
                logger.info("Only including queries that return actual data")
        
        for level in ['easy', 'medium', 'hard']:
            if validate_queries:
                sampled = validate_samples(train_df, level, samples_per_level, random_seed, require_data)
            else:
                sampled = sample_by_difficulty(train_df, level, samples_per_level, random_seed)
            
            if not sampled.empty:
                sampled_dfs.append(sampled)
        
        if not sampled_dfs:
            logger.error("No data could be sampled")
            return False
        
        # Step 3: Combine all sampled data
        final_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Step 4: Save results
        return save_results(final_df, output_dir, output_filename, random_seed)
        
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found")
        logger.error("Tip: Run with --download-data first to download the BookSQL dataset")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data utilities for Text-to-SQL evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --download-data                          # Download BookSQL dataset
  %(prog)s --create-sample                          # Create sample CSV (15 per level)
  %(prog)s --create-sample --samples 5              # Create sample with 5 per level
  %(prog)s --create-sample --validate               # Create sample with SQL validation
  %(prog)s --create-sample --validate --require-data # Only queries that return data
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
        help="Create a balanced sample CSV from BookSQL train.json"
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
        help="Validate SQL queries before including them in the sample"
    )
    
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Only include queries that return actual data (requires --validate)"
    )
    
    args = parser.parse_args()
    
    if args.download_data:
        success = download_booksql_dataset()
        sys.exit(0 if success else 1)
    elif args.create_sample:
        # Validate argument combinations
        if args.require_data and not args.validate:
            logger.error("--require-data requires --validate to be enabled")
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
