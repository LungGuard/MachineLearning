"""
Slice Processor Module

Utilities for preprocessing CT slices and volume data.
Provides 2.5D sandwich creation, center cropping, resizing, and volume cleaning.
Optimized for LIDC-IDRI dataset (pre-calibrated HU).
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging
from DetectionModel.constants.constants.preprocessing import PreProcessingConstants

logger = logging.getLogger(__name__)


class SlicePreprocessor:
    """Utilities for preprocessing CT volume data."""

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
