"""
Diagnostic Script: Analyze Raw CT Volume Values

Run this on a few scans to understand what values we're actually dealing with.
This will help us tune the offset detection threshold properly.

Usage:
    python diagnose_volumes.py

This should be run in your pipeline environment where pylidc is configured.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_volume_for_offset(volume: np.ndarray, patient_id: str = "unknown") -> dict:
    """
    Comprehensive analysis of a CT volume to understand its value distribution.
    """
    # Basic stats on raw volume
    results = {
        "patient_id": patient_id,
        "shape": volume.shape,
        "dtype": str(volume.dtype),
        "raw_min": float(volume.min()),
        "raw_max": float(volume.max()),
        "raw_mean": float(volume.mean()),
        "raw_std": float(volume.std()),
    }
    
    # Check for padding values
    num_very_negative = (volume < -1500).sum()
    num_negative_1000 = ((volume >= -1100) & (volume <= -900)).sum()
    num_around_zero = ((volume >= -100) & (volume <= 100)).sum()
    num_positive = (volume > 100).sum()
    
    total_voxels = volume.size
    results["pct_very_negative"] = 100 * num_very_negative / total_voxels
    results["pct_around_minus1000"] = 100 * num_negative_1000 / total_voxels
    results["pct_around_zero"] = 100 * num_around_zero / total_voxels
    results["pct_positive"] = 100 * num_positive / total_voxels
    
    # Percentiles
    results["p1"] = float(np.percentile(volume, 1))
    results["p5"] = float(np.percentile(volume, 5))
    results["p25"] = float(np.percentile(volume, 25))
    results["p50_median"] = float(np.percentile(volume, 50))
    results["p75"] = float(np.percentile(volume, 75))
    results["p95"] = float(np.percentile(volume, 95))
    results["p99"] = float(np.percentile(volume, 99))
    
    # Center slice analysis (what we use for detection)
    depth = volume.shape[0]
    center_slice = volume[depth // 2]
    
    # Filter padding
    valid_mask = (center_slice > -1500) & (center_slice < 4000)
    valid_pixels = center_slice[valid_mask]
    
    results["center_slice_valid_pixels"] = len(valid_pixels)
    results["center_slice_median"] = float(np.median(valid_pixels)) if len(valid_pixels) > 0 else None
    results["center_slice_mean"] = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else None
    
    # Diagnosis
    median = results["center_slice_median"]
    if median is not None:
        if median > 0:
            results["likely_type"] = "OFFSET_SCAN (median > 0)"
            results["needs_correction"] = True
        elif median > -200:
            results["likely_type"] = "UNCERTAIN (median between -200 and 0)"
            results["needs_correction"] = "MAYBE"
        else:
            results["likely_type"] = "NORMAL_SCAN (median < -200)"
            results["needs_correction"] = False
    else:
        results["likely_type"] = "UNKNOWN (no valid pixels)"
        results["needs_correction"] = "UNKNOWN"
    
    return results


def print_analysis(results: dict):
    """Pretty print the analysis results."""
    print("\n" + "=" * 70)
    print(f"VOLUME ANALYSIS: {results['patient_id']}")
    print("=" * 70)
    
    print(f"\n📐 Shape: {results['shape']}, dtype: {results['dtype']}")
    
    print(f"\n📊 RAW VALUE STATISTICS:")
    print(f"   Min: {results['raw_min']:.1f}")
    print(f"   Max: {results['raw_max']:.1f}")
    print(f"   Mean: {results['raw_mean']:.1f}")
    print(f"   Std: {results['raw_std']:.1f}")
    
    print(f"\n📈 PERCENTILES:")
    print(f"   1%:  {results['p1']:.1f}")
    print(f"   5%:  {results['p5']:.1f}")
    print(f"   25%: {results['p25']:.1f}")
    print(f"   50%: {results['p50_median']:.1f} (median)")
    print(f"   75%: {results['p75']:.1f}")
    print(f"   95%: {results['p95']:.1f}")
    print(f"   99%: {results['p99']:.1f}")
    
    print(f"\n🔍 VALUE DISTRIBUTION:")
    print(f"   Very negative (<-1500, padding): {results['pct_very_negative']:.1f}%")
    print(f"   Around -1000 (air):              {results['pct_around_minus1000']:.1f}%")
    print(f"   Around 0:                        {results['pct_around_zero']:.1f}%")
    print(f"   Positive (>100):                 {results['pct_positive']:.1f}%")
    
    print(f"\n🎯 CENTER SLICE ANALYSIS:")
    print(f"   Valid pixels: {results['center_slice_valid_pixels']}")
    print(f"   Median: {results['center_slice_median']}")
    print(f"   Mean: {results['center_slice_mean']}")
    
    print(f"\n⚡ DIAGNOSIS:")
    print(f"   Type: {results['likely_type']}")
    print(f"   Needs -1024 correction: {results['needs_correction']}")
    
    print("=" * 70)


def run_diagnostics_on_dataset(num_scans: int = 10):
    """
    Run diagnostics on first N scans from your dataset.
    
    This needs to be run in your environment where pylidc is configured.
    """
    import configparser
    configparser.SafeConfigParser = configparser.ConfigParser
    
    # Numpy compatibility
    np.int = np.int64
    np.float = np.float64
    np.bool = np.bool_
    
    import pylidc as pl
    
    print("Querying scans...")
    scans = pl.query(pl.Scan).all()[:num_scans]
    
    all_results = []
    
    for scan in scans:
        patient_id = scan.patient_id
        print(f"\nLoading {patient_id}...")
        
        try:
            volume = scan.to_volume()
            results = analyze_volume_for_offset(volume, patient_id)
            print_analysis(results)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    offset_scans = [r for r in all_results if r.get('needs_correction') == True]
    normal_scans = [r for r in all_results if r.get('needs_correction') == False]
    uncertain = [r for r in all_results if r.get('needs_correction') == "MAYBE"]
    
    print(f"\nOffset scans (need -1024): {len(offset_scans)}")
    for r in offset_scans:
        print(f"   {r['patient_id']}: median={r['center_slice_median']:.1f}")
    
    print(f"\nNormal scans: {len(normal_scans)}")
    for r in normal_scans:
        print(f"   {r['patient_id']}: median={r['center_slice_median']:.1f}")
    
    print(f"\nUncertain: {len(uncertain)}")
    for r in uncertain:
        print(f"   {r['patient_id']}: median={r['center_slice_median']:.1f}")
    
    # Check if threshold 0 would work
    print("\n🔬 THRESHOLD ANALYSIS:")
    all_medians = [(r['patient_id'], r['center_slice_median'], r.get('needs_correction')) 
                   for r in all_results if r['center_slice_median'] is not None]
    
    print("\nAll medians sorted:")
    for pid, med, needs in sorted(all_medians, key=lambda x: x[1]):
        print(f"   {pid}: {med:.1f} (needs_correction={needs})")


if __name__ == "__main__":
    # Run on first 10 scans to understand the data
    run_diagnostics_on_dataset(num_scans=20)