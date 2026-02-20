import cv2 as cv
import numpy as np
import tensorflow as tf
from ClassificationModel.constants.constants.augmentation import AugmenterFields
import tensorflow as tf

class ImageAugmentationPipeline:
    def __init__(self, rotation_range:int=AugmenterFields.DEFAULT_ROTATION_RANGE,
                 flip_probability=AugmenterFields.DEFAULT_FLIP_PROBABILITY,
                 brightness_range=AugmenterFields.DEFAULT_BRIGHTNESS_RANGE,    # ±10% of mean intensity
                 contrast_range=AugmenterFields.DEFAULT_CONTRAST_RANGE,     # ±10% contrast
                 denoise_probability=AugmenterFields.DEFAULT_DENOISE_PROBABILITY):
        self.rotation_range = rotation_range
        self.flip_probability = flip_probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.denoise_probability = denoise_probability

    def __call__(self, image):
        # --- SANITY CHECK (The Fix) ---
        # If the image has no data (size 0) or weird dimensions, return immediately.
        # This prevents OpenCV from crashing on empty images.
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            return image

        # Store original shape to restore later if needed
        original_shape = image.shape
        
        # Ensure we are working with a format OpenCV likes
        image = self._ensure_image_format(image)
        
        # Double check after format ensuring
        if image.size == 0: 
            return image

        og_min = image.min()
        og_max = image.max()
        
        try:
            augmented = self._rotate(image)
            augmented = self._flip(augmented)
            augmented = self._apply_brightness_contrast(augmented, og_max, og_min)
        except cv.error:
            # Fallback: if augmentation fails for any CV reason, return original
            return image.astype(np.float32)

        # Logic to restore dimensions:
        # If the original was 3D (H,W,C) and we accidentally squeezed it to 2D, expand it back.
        if len(original_shape) == 3 and augmented.ndim == 2:
            augmented = np.expand_dims(augmented, axis=-1)
        
        return augmented.astype(np.float32)
        
    def _ensure_image_format(self, image):
        """
        Ensures the image is in a format suitable for OpenCV augmentation.
        """
        img = image.astype(np.float32)
        # Only squeeze if we have a trailing dimension of 1 (e.g. 256,256,1 -> 256,256)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze()
        return img
            
    def _rotate(self, image):
        "rotating the image" 
        # Choosing a random angle
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Robust unpacking (H, W) regardless of channels
        h, w = image.shape[:2] 
        
        # Safety for 0 dimensions (redundant due to main check, but good practice)
        if h == 0 or w == 0:
            return image

        center = (w / 2, h / 2)
        
        # Get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale=1.0)
        
        # Apply rotation
        rotated = cv.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REFLECT
        )

        return rotated
    
    def _flip(self, image):
        should_flip = np.random.random() < self.flip_probability
        return cv.flip(image, 1) if should_flip else image
    
    def _adjust_brightness(self, image):
        brightness_factor = np.random.uniform(
            1 - self.brightness_range, 
            1 + self.brightness_range
        )
        return image * brightness_factor

    def _adjust_contrast(self, image):
        contrast_factor = np.random.uniform(
            1 - self.contrast_range, 
            1 + self.contrast_range
        )
        mean = np.mean(image)
        adjusted = (image - mean) * contrast_factor + mean
        return adjusted
    
    def _apply_brightness_contrast(self, image, max_val, min_val):
        augmented = self._adjust_brightness(image)
        augmented = self._adjust_contrast(augmented)
        return np.clip(augmented, min_val, max_val)
    
    def _denoise(self, image):
        should_denoise = np.random.random() < self.denoise_probability
        denoised = cv.bilateralFilter(
            image,
            d=5,              
            sigmaColor=30,    
            sigmaSpace=30     
        ) if should_denoise else image
        return denoised
def apply_augmentation(split, augmenter):
    if not augmenter:
        raise ValueError("Augmenter cannot be None")

    # Use tf.function for better performance
    @tf.function
    def augment_map(image, label):
        def augment_wrapper(img):
            return augmenter(img)
        
        augmented = tf.numpy_function(
            func=augment_wrapper,
            inp=[image],
            Tout=tf.float32
        )
        # Critical: Set shape explicitly after numpy_function
        augmented.set_shape(image.shape)
        return augmented, label
    
    augmented_dataset = (
        split
        .map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return augmented_dataset