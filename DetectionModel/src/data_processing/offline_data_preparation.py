"""
Unified Data Preparation Pipeline.
Contains the shared logic for both Serial and Parallel processing.
"""

import os
import sys
import logging
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from .config import DataPrepConfig
from .pylidc_config import configure_pylidc, import_pylidc
from .data_splitter import split_patients_by_id, get_patient_split
from .MONAI_scan_processor import CTScanProcessor as MonaiProcessor
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)
from .diagnose_dataset import DatasetDiagnoser


logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False) -> None:
    """Configures global logging settings."""
    log_level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('pylidc').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def create_directory_structure(output_dir: str) -> Dict[str, Path]:
    """Creates the YOLO directory structure."""
    base_path = Path(output_dir)
    directories = {
        'train_images': base_path / 'train' / 'images',
        'train_labels': base_path / 'train' / 'labels',
        'val_images': base_path / 'val' / 'images',
        'val_labels': base_path / 'val' / 'labels',
        'test_images': base_path / 'test' / 'images',
        'test_labels': base_path / 'test' / 'labels',
        'metadata': base_path / 'metadata'
    }

    for name, path in directories.items():
        if path.exists() and name != 'metadata':
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        if name == 'metadata':
            for f in path.iterdir():
                if f.is_file() and not f.name.startswith("patient_splits"):
                    f.unlink()
    return directories


def filter_scans_with_nodules(all_scans: List, pylidc_module) -> List[Tuple]:
    """Filters scans that actually contain clustered annotations."""
    total = len(all_scans)
    logger.info(f"Filtering {total} scans for nodules...")
    scans_with_nodules = []

    for idx, scan in enumerate(all_scans):
        try:
            if len(scan.cluster_annotations()) > 0:
                scans_with_nodules.append((scan, scan.cluster_annotations()))
        except Exception:
            pass
        if (idx + 1) % 100 == 0:
            logger.info(f" Checked {idx + 1}/{total} scans...")

    logger.info(f"Found {len(scans_with_nodules)} scans with valid nodules.")
    return scans_with_nodules


class DataPreparationPipeline:
    """
    Base class that handles the setup, configuration, processing (serial), and interactive cleanup.
    """
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.pylidc = None
        self.directories = {}
        self.scans_to_process = []
        self.splits = {}

    def setup(self):
        """Runs the initialization steps."""
        logger.info("=" * 50)
        logger.info("LungGuard Data Preparation Pipeline Setup")
        logger.info("=" * 50)

        logger.info("[1/4] Configuring PyLIDC...")
        configure_pylidc(self.config.data_path)
        self.pylidc = import_pylidc()

        logger.info("[2/4] Creating directories...")
        self.directories = create_directory_structure(self.config.output_dir)

        self._prepare_splits_and_scans()

    def _prepare_splits_and_scans(self):
        splits_json_path = self.directories['metadata'] / 'patient_splits.json'
        
        if splits_json_path.exists():
            logger.info("[3/4] Loading existing patient splits...")
            with open(splits_json_path, 'r') as f:
                patient_splits_dict = json.load(f)
            
            self.splits = {'train': [], 'val': [], 'test': []}
            for pid, split in patient_splits_dict.items():
                self.splits[split].append(pid)
                
            patient_ids = list(patient_splits_dict.keys())
            logger.info(f"Querying scans for {len(patient_ids)} existing patients...")
            
            for pid in patient_ids:
                try:
                    scans = self.pylidc.query(self.pylidc.Scan).filter(self.pylidc.Scan.patient_id == pid).all()
                    for s in scans:
                        if len(s.cluster_annotations()) > 0:
                            self.scans_to_process.append((s, s.cluster_annotations()))
                except: pass
        else:
            logger.info("[3/4] Querying full database and creating new splits...")
            all_scans = self.pylidc.query(self.pylidc.Scan).all()
            self.scans_to_process = filter_scans_with_nodules(all_scans, self.pylidc)
            
            if not self.scans_to_process:
                raise ValueError("No valid scans found in database!")

            patient_ids = list(set(s.patient_id for s, _ in self.scans_to_process))
            self.splits = split_patients_by_id(
                patient_ids, self.config.train_ratio, self.config.val_ratio, 
                self.config.test_ratio, self.config.random_seed
            )
            
            patient_splits_dict = {}
            for pid in patient_ids:
                patient_splits_dict[pid] = get_patient_split(pid, self.splits)
            
            with open(splits_json_path, 'w') as f:
                json.dump(patient_splits_dict, f, indent=2, sort_keys=True)

        logger.info(f"Ready to process {len(self.scans_to_process)} scans.")

    def finalize(self, all_metadata: List[Dict], extra_config_info: Dict = None) -> Path:
        """Saves CSV, JSON, YAML and logs stats."""
        logger.info("[Finalizing] Saving outputs...")
        csv_path = self.directories['metadata'] / 'regression_dataset.csv'
        metadata_df = save_metadata_csv(all_metadata, csv_path)
        config_path = self.directories['metadata'] / 'preparation_config.json'
        config_dict_full = save_config_json(self.config, config_path, metadata_df)
        if extra_config_info:
            config_dict_full.update(extra_config_info)
        yaml_path = save_yolo_yaml(self.config.output_dir, metadata_df)
        log_summary_statistics(metadata_df, config_dict_full, csv_path, config_path, yaml_path)
        return csv_path

    def run_serial(self) -> Path:
        """Runs serial processing and then triggers interactive cleanup."""
        self.setup()
        processor = MonaiProcessor(self.config, self.directories)
        all_metadata = []
        logger.info(f"Starting Serial Processing of {len(self.scans_to_process)} scans...")
        
        for idx, (scan, _) in enumerate(self.scans_to_process):
            split = get_patient_split(scan.patient_id, self.splits)
            if (idx + 1) % self.config.log_freq == 0:
                logger.info(f"Processing {idx+1}/{len(self.scans_to_process)}: {scan.patient_id}")
            try:
                meta = processor.process_scan(scan, split, pl_module=self.pylidc)
                all_metadata.extend(meta)
            except Exception as e:
                logger.error(f"Error processing {scan.patient_id}: {e}")

        csv_path = self.finalize(all_metadata, {"mode": "serial"})
        
        # Trigger Interactive Cleanup at the end
        self.interactive_cleanup()
        
        return csv_path

    def interactive_cleanup(self):
        """
        Runs diagnosis and offers the user to save reports or export a clean dataset.
        """
        logger.info("\n" + "="*60)
        logger.info("POST-PROCESSING: DATASET DIAGNOSIS & CLEANUP")
        logger.info("="*60)
        
        diagnoser = DatasetDiagnoser(self.config.output_dir)
        logger.info("Running analysis on generated images... Please wait.")
        diagnoser.analyze()
        
        summary = diagnoser.get_summary_report()
        print(f"\nAnalysis Complete.")
        print(f"Total Images: {summary['total_images']}")
        print(f"Problematic Images: {summary['problematic_count']} ({summary['problematic_ratio']:.1%})")

        print("\n" + "-"*30)
        save_ans = input(">> Do you want to save detailed analysis CSV reports? (y/n): ").strip().lower()
        if save_ans == 'y':
            diagnoser.save_reports_to_disk()
            print("Reports saved.")

        # 3. Ask to Export/Clean Dataset
        print("\n" + "-"*30)
        print("Clean Dataset Export Options:")
        print("1. Create NEW clean dataset (Copy valid files to a new folder)")
        print("2. Clean IN-PLACE (DELETE invalid files from current folder)")
        print("3. Skip")
        
        choice = input(">> Select an option (1/2/3): ").strip()
        
        if choice == '1':
            new_dir_name = input(">> Enter name/path for the new dataset folder: ").strip()
            if new_dir_name:
                # Ensure it's a full path or relative to current dir
                new_path = Path(new_dir_name).resolve()
                print(f"Exporting clean dataset to: {new_path} ...")
                diagnoser.export_clean_dataset(output_dir=str(new_path), overwrite_existing=False)
        
        elif choice == '2':
            confirm = input(">> WARNING: This will PERMANENTLY DELETE bad files. Type 'yes' to confirm: ").strip()
            if confirm == 'yes':
                print("Cleaning dataset in-place...")
                diagnoser.export_clean_dataset(overwrite_existing=True)
            else:
                print("Operation cancelled.")
        
        print("\nPipeline execution finished.")