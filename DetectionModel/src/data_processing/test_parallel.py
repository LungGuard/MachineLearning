"""Test script for parallel data preparation pipeline."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from DetectionModel.src.data_processing.parallel_data_preparation import main_parallel


def test_parallel_pipeline():
    """
    Test the parallel data preparation pipeline with a small subset.
    
    This will help validate:
    1. The multiprocessing implementation works correctly
    2. Results are comparable to sequential version
    3. Performance improvement is measurable
    """
    
    print("=" * 70)
    print("Testing Parallel Data Preparation Pipeline")
    print("=" * 70)
    
    # Test configuration
    config_overrides = {
        'data_path': 'D:/LIDC-IDRI',  # Update this to your actual path
        'output_dir': './test_parallel_output',
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'min_diameter': 3.0,
        'max_diameter': 100.0,
        'slices_per_nodule': 3,
        'seed': 42,
        'debug': False,  # Set True for verbose logging
        'log_freq': 1,  # Log every scan for testing
        'num_workers': 4,  # Start with 4 workers
    }
    
    print(f"\nConfiguration:")
    print(f"  Data path: {config_overrides['data_path']}")
    print(f"  Output dir: {config_overrides['output_dir']}")
    print(f"  Workers: {config_overrides['num_workers']}")
    print(f"  Debug mode: {config_overrides['debug']}")
    
    # Run the parallel pipeline
    print("\n" + "=" * 70)
    print("Starting parallel processing...")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        result_path = main_parallel(config_overrides)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✓ Pipeline completed successfully!")
        print("=" * 70)
        print(f"  Result saved to: {result_path}")
        print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Load and display summary
        import pandas as pd
        df = pd.read_csv(result_path)
        
        print(f"\n  Dataset Summary:")
        print(f"    Total samples: {len(df)}")
        print(f"    Train: {len(df[df['split'] == 'train'])}")
        print(f"    Val: {len(df[df['split'] == 'val'])}")
        print(f"    Test: {len(df[df['split'] == 'test'])}")
        print(f"    Unique patients: {df['patient_id'].nunique()}")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✗ Pipeline failed!")
        print("=" * 70)
        print(f"  Error: {e}")
        print(f"  Time before failure: {elapsed_time:.2f} seconds")
        raise


if __name__ == "__main__":
    print("\n")
    print("IMPORTANT: Update 'data_path' in this script before running!")
    print("Current path: 'D:/LIDC-IDRI'")
    print("\n")
    
    response = input("Have you updated the data path? (yes/no): ").strip().lower()
    
    if response == 'yes':
        test_parallel_pipeline()
    else:
        print("\nPlease update the 'data_path' in test_parallel.py first.")
        print("Then run: python DetectionModel/src/data_processing/test_parallel.py")
