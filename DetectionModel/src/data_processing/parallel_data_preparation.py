"""Parallel offline data preparation pipeline using multiprocessing."""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

# Imports
from .offline_data_preparation import (
    DataPreparationPipeline, 
    setup_logging, 
    import_pylidc, 
    configure_pylidc,
    DataPrepConfig,
    MonaiProcessor,
    get_patient_split
)

logger = logging.getLogger(__name__)

# Global worker variable
_worker_pylidc = None

def _init_worker(data_path: str):
    """Initialize each worker process with its own pylidc configuration."""
    global _worker_pylidc
    try:
        configure_pylidc(data_path)
        _worker_pylidc = import_pylidc()
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        _worker_pylidc = None

def _worker_process_scan(args: Tuple) -> List[Dict]:
    """Worker function to process a single scan."""
    patient_id, patient_split, config_dict, directories_dict = args
    global _worker_pylidc
    pylidc = _worker_pylidc

    config = DataPrepConfig(**config_dict)
    directories = {k: Path(v) for k, v in directories_dict.items()}

    try:
        scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == patient_id).first()
        if not scan: return []

        processor = MonaiProcessor(config, directories)
        scan_metadata = processor.process_scan(scan, patient_split, pl_module=pylidc)
        return scan_metadata

    except Exception as e:
        logger.error(f"[{patient_id}] Worker error: {e}")
        return []

def run_parallel_pipeline(config: DataPrepConfig, num_workers: int = None) -> Path:
    """Orchestrates the parallel data preparation."""
    
    # 1. Initialize shared pipeline logic
    pipeline = DataPreparationPipeline(config)
    pipeline.setup()

    # 2. Prepare arguments for workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    logger.info(f"[Parallel] Starting processing with {num_workers} workers...")
    
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    directories_dict = {k: str(v) for k, v in pipeline.directories.items()}
    
    task_args = []
    for scan, _ in pipeline.scans_to_process:
        pid = scan.patient_id
        split = get_patient_split(pid, pipeline.splits)
        task_args.append((pid, split, config_dict, directories_dict))

    # 3. Run Multiprocessing
    all_metadata = []
    completed = 0
    total = len(task_args)

    try:
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(config.data_path,)) as pool:
            for result in pool.imap_unordered(_worker_process_scan, task_args, chunksize=1):
                completed += 1
                if result:
                    all_metadata.extend(result)
                
                if completed % config.log_freq == 0:
                    logger.info(f"  [Parallel] Progress: {completed}/{total} scans processed.")
                    
    except KeyboardInterrupt:
        logger.warning("Interrupted! Stopping workers...")
        pool.terminate()
        pool.join()
        raise

    csv_path = pipeline.finalize(all_metadata, {"num_workers": num_workers})
    
    pipeline.interactive_cleanup()
    
    return csv_path


def main_parallel(config_overrides: Dict = None, num_workers: int = None):
    # Manual default config (since we moved PARALLEL_DEFAULT_CONFIG)
    config_values = {
        'data_path': r"E:\FinalsProject\Datasets\CancerDetection\images\manifest-1600709154662\LIDC-IDRI",
        'output_dir': r".\DetectionModel\datasets_monai",
        'train_ratio': 0.70, 'val_ratio': 0.15, 'test_ratio': 0.15,
        'min_diameter': 3.0, 'max_diameter': 100.0,
        'slices_per_nodule': 3, 'seed': 42, 'debug': False, 'log_freq': 5
    }
    
    if config_overrides: 
        config_values.update(config_overrides)
        
    setup_logging(debug=config_values['debug'])

    config = DataPrepConfig(
        data_path=config_values['data_path'],
        output_dir=config_values['output_dir'],
        train_ratio=config_values['train_ratio'],
        val_ratio=config_values['val_ratio'],
        test_ratio=config_values['test_ratio'],
        min_nodule_diameter=config_values['min_diameter'],
        max_nodule_diameter=config_values['max_diameter'],
        slices_per_nodule=config_values['slices_per_nodule'],
        random_seed=config_values['seed'],
        log_freq=config_values['log_freq']
    )

    return run_parallel_pipeline(config, num_workers=num_workers)


if __name__ == "__main__":
    config_overrides = {
        'output_dir': r".\DetectionModel\datasets_monai",
        # 'num_workers': 4 
    }
    main_parallel(config_overrides)