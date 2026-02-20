"""
Slice Processor Module

Utilities for preprocessing CT slices and volume data.
Provides 2.5D sandwich creation, center cropping, resizing, and volume cleaning.
Optimized for LIDC-IDRI dataset (pre-calibrated HU).
"""

import numpy as np
import cv2
from scipy.ndimage import zoom
from typing import Tuple, Optional
import logging
from constants.detection.preprocessing_constants import HUValues,PreProcessingConstants

logger = logging.getLogger(__name__)


class SlicePreprocessor:
    """Utilities for preprocessing CT volume data."""

    @staticmethod
    def clean_and_fix_volume(volume: np.ndarray) -> np.ndarray:
        """
        Clean the volume data for processing with offset detection and padding handling.

        Operations (in order):
        1. Cast to float32 for precision
        2. Handle NaNs (replace with air = -1000 HU)
        3. Create padding mask (values < -1500 are padding like -2048, -3024)
        4. Detect offset scans (if 5th percentile of valid pixels > -100, apply -1024 offset)
        5. Set padding to exactly -1000 HU (air) BEFORE resampling to prevent interpolation artifacts
        6. Clip to valid HU range [-1000, 3000]

        Some LIDC-IDRI scans have:
        - Offset values (air at 0 HU instead of -1000 HU)
        - Padding values that corrupt interpolation during resampling

        Returns:
            Cleaned volume ready for resampling
        """
        volume = volume.astype(np.float32)

        nan_count = np.isnan(volume).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, replacing with -1000")
            volume = np.nan_to_num(volume, nan=HUValues.AIR_HU)

        # 3. Create padding mask (values < -1500 are typically padding: -2048, -3024, etc.)
        PADDING_THRESHOLD = -1500.0
        padding_mask = volume < PADDING_THRESHOLD
        padding_voxel_count = padding_mask.sum()

        if padding_voxel_count > 0:
            logger.debug(f"Found {padding_voxel_count} padding voxels (< {PADDING_THRESHOLD})")

        # 4. Detect offset scans using center slice
        # Sample the center slice to avoid edge artifacts
        depth = volume.shape[0]
        center_slice = volume[depth // 2]

        # Create valid mask for sampling (exclude padding and extreme values)
        valid_mask = (center_slice > PADDING_THRESHOLD) & (center_slice < 4000)
        valid_pixels = center_slice[valid_mask]

        offset_applied = False
        if len(valid_pixels) > 0:
            # Use 5th percentile to detect offset
            # In a normal scan, 5th percentile should be around -900 to -1000 (air/lung)
            # In an offset scan, 5th percentile would be around 0 to 100
            percentile_5 = np.percentile(valid_pixels, 5)

            OFFSET_DETECTION_THRESHOLD = -100.0

            if percentile_5 > OFFSET_DETECTION_THRESHOLD:
                logger.info(
                    f"Offset scan detected (5th percentile={percentile_5:.1f} > {OFFSET_DETECTION_THRESHOLD}). "
                    f"Applying -1024 correction."
                )
                # Apply offset correction to NON-padding voxels only
                volume[~padding_mask] -= 1024.0
                offset_applied = True
            else:
                logger.debug(f"Normal scan (5th percentile={percentile_5:.1f})")
        else:
            logger.warning("No valid pixels found in center slice for offset detection")

        # 5. Set padding to exactly -1000 HU (air) BEFORE resampling
        # This prevents extreme values from corrupting interpolation
        if padding_voxel_count > 0:
            volume[padding_mask] = HUValues.AIR_HU  # -1000
            logger.debug(f"Set {padding_voxel_count} padding voxels to {HUValues.AIR_HU} HU")

        # 6. Clip to valid HU range (safety measure)
        volume = np.clip(volume, -1000.0, 3000.0)

        logger.debug(
            f"Volume cleaned: min={volume.min():.1f}, max={volume.max():.1f}, "
            f"mean={volume.mean():.1f}, offset_applied={offset_applied}"
        )

        return volume
    
    @staticmethod
    def resample_volume(
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        Resample a 3D CT volume to isotropic voxel spacing.
        
        Medical CT scans have non-uniform voxel spacing (e.g., 0.7mm x 0.7mm x 2.5mm).
        This function resamples to uniform spacing for consistent analysis.
        """
        volume = volume.astype(np.float32)
        
        zoom_factors = tuple(
            orig / target 
            for orig, target in zip(original_spacing, target_spacing)
        )
        
        resampled = zoom(volume, zoom_factors, order=1, mode='nearest')
        
        logger.debug(
            f"Resampled volume from {volume.shape} to {resampled.shape} "
            f"(spacing: {original_spacing} -> {target_spacing})"
        )
        
        return resampled
    
    @staticmethod
    def apply_windowing(
        volume: np.ndarray,
        center: float = -600.0,
        width: float = 1500.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply CT windowing for lung visualization.
        
        Standard lung window: center=-600, width=1500
        This maps HU range [-1350, +150] to display values [0, 1]
        
        Expected mapping for LIDC-IDRI (already calibrated):
        - Air (-1000 HU) -> 0.23 (dark grey, almost black)
        - Lung tissue (-700 to -500 HU) -> 0.43 to 0.57 (medium grey)
        - Soft tissue (0 HU) -> 0.90 (light grey)
        - Bone (+400 HU) -> 1.0 (white, clipped)
        
        Note: Air maps to ~0.23, not 0.0, because the window range starts at -1350.
        This is correct and expected for lung windowing.
        """
        lower = center - (width / 2.0)  # -1350 HU
        upper = center + (width / 2.0)  # +150 HU

        windowed = np.interp(volume, [lower, upper], [0.0, 1.0])
        
        logger.debug(
            f"Windowing applied: center={center}, width={width}, "
            f"result range=[{windowed.min():.3f}, {windowed.max():.3f}]"
        )

        return windowed.astype(np.float32)

    @staticmethod
    def resize_slice_to_target(
        slice_2d: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        preserve_aspect_ratio: bool = True,
        pad_value: float = 0.0
    ) -> np.ndarray:
        """
        Resize a 2D slice to a fixed target size.

        Args:
            slice_2d: Input 2D array (H, W) or (H, W, C) for RGB
            target_size: Target dimensions as (height, width)
            preserve_aspect_ratio: If True, preserve aspect ratio and pad with pad_value
            pad_value: Value to use for padding (0 = black/air after windowing)

        Returns:
            Resized slice with shape (target_size[0], target_size[1]) or
            (target_size[0], target_size[1], C) for RGB input
        """
        target_h, target_w = target_size
        is_rgb = len(slice_2d.shape) == 3
        h, w = slice_2d.shape[:2]

        if not preserve_aspect_ratio:
            return cv2.resize(
                slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        # Preserve aspect ratio with padding
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(slice_2d, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded output
        if is_rgb:
            output = np.full((target_h, target_w, slice_2d.shape[2]), pad_value, dtype=slice_2d.dtype)
        else:
            output = np.full((target_h, target_w), pad_value, dtype=slice_2d.dtype)

        # Center the resized image in the output
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        if is_rgb:
            output[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
        else:
            output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        logger.debug(
            f"Resized slice from ({h}, {w}) to ({target_h}, {target_w}) "
            f"with aspect preservation (scaled to {new_h}x{new_w}, padded)"
        )

        return output

    @staticmethod
    def center_crop_slice(
        slice_2d: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize so the smallest dimension fills the target, then center crop.

        Returns:
            Tuple of:
            - output_image: (target_h, target_w) or (target_h, target_w, C)
            - scale: the scale factor applied
            - effective_offset: (y_offset, x_offset) — always >= 0 (crop only)
        """
        target_h, target_w = target_size
        h, w = slice_2d.shape[:2]
        is_rgb = len(slice_2d.shape) == 3

        natural_scale = max(target_h / h, target_w / w)
        scale = min(natural_scale, PreProcessingConstants.MAX_CROP_SCALE)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized = cv2.resize(slice_2d, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create output canvas
        if is_rgb:
            output = np.zeros((target_h, target_w, slice_2d.shape[2]), dtype=slice_2d.dtype)
        else:
            output = np.zeros((target_h, target_w), dtype=slice_2d.dtype)

        # Calculate offsets for centering
        # If resized > target: positive offset (we crop from resized)
        # If resized < target: negative offset (we pad in output)
        y_diff = new_h - target_h
        x_diff = new_w - target_w

        # Source region in resized image
        src_y_start = max(0, y_diff // 2)
        src_x_start = max(0, x_diff // 2)
        src_y_end = src_y_start + min(new_h, target_h)
        src_x_end = src_x_start + min(new_w, target_w)

        # Destination region in output
        dst_y_start = max(0, -y_diff // 2)
        dst_x_start = max(0, -x_diff // 2)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy resized image to output
        if is_rgb:
            output[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                resized[src_y_start:src_y_end, src_x_start:src_x_end, :]
        else:
            output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                resized[src_y_start:src_y_end, src_x_start:src_x_end]

        # Effective offset for bbox adjustment:
        # - crop_offset in resized image space (positive = cropped)
        # - pad_offset in output space (negative = padded)
        # We return (crop_y, crop_x) where positive means cropped, negative means padded
        effective_y_offset = y_diff // 2  # Positive if cropped, negative if padded
        effective_x_offset = x_diff // 2

        logger.debug(
            f"Center crop: ({h}, {w}) -> scale {scale:.3f} -> "
            f"({new_h}, {new_w}) -> offset ({effective_y_offset}, {effective_x_offset}) -> "
            f"({target_h}, {target_w})"
        )

        return output, scale, (effective_y_offset, effective_x_offset)

    @staticmethod
    def create_25d_sandwich(
        volume: np.ndarray,
        z_index: int,
        target_size: Optional[Tuple[int, int]] = None,
        use_center_crop: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[float, Tuple[int, int]]]]:
        """
        Create a 2.5D RGB image from adjacent CT slices.

        Channels: R=z-1, G=z, B=z+1

        Returns:
            Tuple of:
            - RGB image as uint8 (H, W, 3)
            - crop_info: (scale, (y_offset, x_offset)) if center_crop used, None otherwise
        """
        depth, height, width = volume.shape

        z_prev = max(0, z_index - 1)
        z_curr = z_index
        z_next = min(depth - 1, z_index + 1)

        sandwich = np.stack([volume[z_prev], volume[z_curr], volume[z_next]], axis=-1)
        rgb_image = (sandwich * 255.0).astype(np.uint8)

        crop_info = None

        if target_size is not None:
            if use_center_crop:
                rgb_image, scale, crop_offset = SlicePreprocessor.center_crop_slice(
                    rgb_image, target_size
                )
                crop_info = (scale, crop_offset)
            else:
                rgb_image = SlicePreprocessor.resize_slice_to_target(
                    rgb_image,
                    target_size=target_size,
                    preserve_aspect_ratio=True,
                    pad_value=0
                )

        return rgb_image, crop_info