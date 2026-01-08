import cv2 as cv
import numpy as np
from constants.classification.augmentation_constants import AugmenterFields

class ImageAugmentationPipeline:
    def __init__(self,rotation_range:int=AugmenterFields.DEFAULT_ROTATION_RANGE,
                 flip_probability=AugmenterFields.DEFAULT_FLIP_PROBABILITY,
                 brightness_range=AugmenterFields.DEFAULT_BRIGHTNESS_RANGE,    # ±10% of mean intensity
                 contrast_range=AugmenterFields.DEFAULT_CONTRAST_RANGE,     # ±10% contrast
                 denoise_probability=AugmenterFields.DEFAULT_DENOISE_PROBABILITY):
        self.rotation_range=rotation_range
        self.flip_probability = flip_probability
        self.brightness_range=brightness_range
        self.contrast_range=contrast_range
        self.denoise_probability=denoise_probability


    def __call__(self, image):
        image = self._ensure_image_dim(image)
        
        og_min = image.min()
        og_max = image.max()
        
        augmented = self._rotate(image)
        augmented = self._flip(augmented)
        augmented = self._apply_brightness_contrast(augmented,og_max,og_min)

        return augmented
        
    
    def _ensure_image_dim(self,image):
        " ensuring that the image is 2d for the augmentation "

        img = image.squeeze() if len(image.shape) == 3 else image
        img = img.astype(np.float32)
        return img
    
    def _rotate(self,image):
        "rotating the image" 
        #choosing a random angle for the rotation from the rotation range
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape
        center = (w / 2, h / 2)
        
        #get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale=1.0)
        
        #apply rotation
        rotated = cv.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv.INTER_CUBIC,      # Smooth interpolation
            borderMode=cv.BORDER_REFLECT  # Natural edge handling
        )

        return rotated
    
    def _flip(self,image):
        should_flip = np.random.random() < self.flip_probability

        return cv.flip(image, 1) if should_flip else image
    
    def _adjust_brightness(self,image):
        
        brightness_factor = np.random.uniform(
            1 - self.brightness_range, 
            1 + self.brightness_range
        )

        return image*brightness_factor
    def _adjust_contrast(self,image):
        contrast_factor = np.random.uniform(
            1 - self.contrast_range, 
            1 + self.contrast_range
        )
        mean = np.mean(image)
        adjusted = (image - mean) * contrast_factor + mean
        return adjusted
    
    def _apply_brightness_contrast(self,image,max,min):
        augmented=self._adjust_brightness(image)
        augmented=self._adjust_contrast(augmented)
        
        # Clipping to original range to prevent invalid values
        return np.clip(augmented, min, max)
    
    def _denoise(self, image):
        "denoising images - Optional"
        should_denoise = np.random.random() < self.denoise_probability
        
        denoised = cv.bilateralFilter(
            image,
            d=5,              # Diameter of pixel neighborhood
            sigmaColor=30,    # Filter sigma in color space
            sigmaSpace=30     # Filter sigma in coordinate space
        ) if should_denoise else image
        
        return denoised

