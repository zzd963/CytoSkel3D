import os
import numpy as np
from skimage import io, filters, morphology, exposure, feature, util, img_as_ubyte, img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter, zoom, binary_closing, distance_transform_edt, convolve
import matplotlib
from datetime import datetime
import tifffile
from skimage.transform import rotate
from skimage import measure
from scipy.spatial import cKDTree

class BinaryProcessor:
    """Independent binarization processor class"""

    def __init__(self, params: dict, im_info):
        self.params = params
        self.im_info = im_info
        self.raw_data = self.im_info.get_memmap('processed')  # Raw data, read-only
        self.is_3d = self.raw_data.ndim == 3  # Automatically detect dimensions

    def _log_transform(self, image):
        """Apply logarithmic transformation"""
        min_val = np.min(image)
        if min_val <= 0:
            offset = 1 - min_val
        else:
            offset = 0

        log_image = np.log(image + offset)
        return log_image, {"offset": offset, "original_min": min_val}

    def _inverse_log_transform(self, threshold, conversion_dict):
        """Reverse logarithmic transformation"""
        offset = conversion_dict["offset"]
        original_min = conversion_dict["original_min"]

        exp_threshold = np.exp(threshold) - offset
        # Ensure threshold does not fall below original minimum
        exp_threshold = np.maximum(exp_threshold, original_min)
        return exp_threshold

    def _get_global_threshold(self, image):
        """Calculate global threshold"""
        method = self.params['binary_method']
        min_val = self.params.get('threshold_min', 0)
        max_val = self.params.get('threshold_max', 1)
        correction_factor = self.params.get('threshold_correction_factor', 1)
        assign_middle = self.params.get('assign_middle_to_foreground', "foreground")
        log_transform = self.params.get('log_transform', False)

        # Apply logarithmic transformation
        if log_transform:
            image, conversion_dict = self._log_transform(image)

        # Handle case where all pixel values are the same
        if np.all(image == image.flat):
            threshold = image.flat
            if log_transform:
                threshold = self._inverse_log_transform(threshold, conversion_dict)
            threshold = min(max(threshold, min_val), max_val)
            return threshold * correction_factor

        # Calculate threshold based on method
        if method == 'otsu':
            threshold = filters.threshold_otsu(image)
        elif method == 'multiotsu':
            classes = self.params.get('multiotsu_classes', 3)
            thresholds = filters.threshold_multiotsu(image, classes=classes)
            # Choose whether middle value is foreground or background
            if assign_middle.casefold() == "foreground":
                threshold = thresholds  # First threshold as foreground boundary
            else:
                threshold = thresholds  # Second threshold as foreground boundary
        elif method == 'minimum_cross_entropy':
            # Corresponds to Li's method in skimage
            threshold = filters.threshold_li(image)

        # Reverse logarithmic transformation and correct
        if log_transform:
            threshold = self._inverse_log_transform(threshold, conversion_dict)

        threshold = min(max(threshold, min_val), max_val)
        return threshold * correction_factor

    def _get_adaptive_threshold(self, image):
        """Calculate adaptive threshold map"""
        method = self.params['binary_method']
        window_size = self.params.get('adaptive_block_size', 15)
        min_val = self.params.get('threshold_min', 0)
        max_val = self.params.get('threshold_max', 1)
        correction_factor = self.params.get('threshold_correction_factor', 1)
        assign_middle = self.params.get('assign_middle_to_foreground', "foreground")
        global_limits = self.params.get('global_limits', [0.7, 1.5])
        log_transform = self.params.get('log_transform', False)

        # Apply logarithmic transformation
        if log_transform:
            image, conversion_dict = self._log_transform(image)

        # Get global threshold as reference
        global_threshold = self._get_global_threshold(image)

        # Calculate adaptive threshold map
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd

        # Calculate threshold for each block
        image_size = np.array(image.shape[:2], dtype=int)
        nblocks = image_size // window_size
        if any(n < 2 for n in nblocks):
            raise ValueError(
                f"Adaptive window size {window_size}px is too large "
                f"to be used for {image_size}x{image_size} image"
            )

        # Calculate block size and position
        increment = np.array(image_size, dtype=float) / np.array(nblocks, dtype=float)
        threshold_map = np.zeros(image_size, dtype=float)
        block_thresholds = np.zeros([nblocks, nblocks])

        # Calculate threshold for each block
        for i in range(nblocks):
            i0 = int(i * increment)
            i1 = int((i + 1) * increment)
            for j in range(nblocks):
                j0 = int(j * increment)
                j1 = int((j + 1) * increment)
                block = image[i0:i1, j0:j1]

                # Skip empty blocks
                if np.size(block) == 0:
                    block_threshold = 0
                # Handle case where all pixel values are the same
                elif np.all(block == block.flat):
                    block_threshold = block.flat
                # Handle special case for multi-Otsu method
                elif method == 'multiotsu' and len(np.unique(block)) < 3:
                    block_threshold = filters.threshold_otsu(block)
                else:
                    # Calculate threshold within the block
                    if method == 'otsu':
                        block_threshold = filters.threshold_otsu(block)
                    elif method == 'multiotsu':
                        classes = self.params.get('multiotsu_classes', 3)
                        thresholds = filters.threshold_multiotsu(block, classes=classes)
                        block_threshold = thresholds if assign_middle.casefold() == "foreground" else thresholds
                    elif method == 'minimum_cross_entropy':
                        block_threshold = filters.threshold_li(block)

                block_thresholds[i, j] = block_threshold

        # Smooth threshold map using bilinear interpolation
        x_coords = np.arange(0, nblocks) * increment + increment / 2
        y_coords = np.arange(0, nblocks) * increment + increment / 2

        from scipy.interpolate import RectBivariateSpline
        spline = RectBivariateSpline(y_coords, x_coords, block_thresholds)

        y_points = np.arange(0, image_size)
        x_points = np.arange(0, image_size)
        threshold_map = spline(y_points, x_points, grid=True)

        # Reverse logarithmic transformation
        if log_transform:
            threshold_map = self._inverse_log_transform(threshold_map, conversion_dict)

        # Apply correction factor and global limits
        threshold_map *= correction_factor
        t_min = max(min_val, global_threshold * global_limits)
        t_max = min(max_val, global_threshold * global_limits)
        np.clip(threshold_map, t_min, t_max, out=threshold_map)

        return threshold_map

    def process(self, image: np.ndarray) -> np.ndarray:
        """Execute binarization processing"""
        strategy = self.params.get('binary_strategy', 'Global')
        min_noise_size = self.params.get('min_noise_size', 10)
        morph_radius = self.params.get('morph_radius', 1)
        apply_morphology = self.params.get('apply_morphology', True)

        # Select binarization strategy
        if strategy == 'Global':
            # Use single threshold for the entire image
            threshold = self._get_global_threshold(image)
            binary = image > threshold
        else:  # Adaptive
            # Process 3D image layer by layer
            if image.ndim == 3:
                adaptive_threshold = np.zeros_like(image, dtype=np.float32)
                for z in range(image.shape):
                    adaptive_threshold[z] = self._get_adaptive_threshold(image[z])
                binary = image > adaptive_threshold
            else:
                threshold_map = self._get_adaptive_threshold(image)
                binary = image > threshold_map

        # Morphological operations
        if apply_morphology:
            if image.ndim == 3:
                selem = morphology.ball(morph_radius)
            else:
                selem = morphology.disk(morph_radius)

            binary = morphology.binary_closing(binary, selem)
            binary = morphology.binary_opening(binary, selem)

        # Remove small objects
        connectivity = 3 if image.ndim == 3 else 2
        binary = morphology.remove_small_objects(
            binary,
            min_size=min_noise_size,
            connectivity=connectivity
        )
        binary = morphology.remove_small_holes(
            binary,
            area_threshold=8,
            connectivity=3 if self.is_3d else 2
        )

        return binary.astype(bool)