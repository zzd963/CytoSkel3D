import os
import numpy as np
from skimage import io, filters, morphology, exposure, feature, util, img_as_ubyte, img_as_float
from scipy.ndimage import gaussian_filter, zoom, binary_closing, distance_transform_edt, convolve, generic_filter
from skimage import measure
from itertools import product
import warnings


class SkeletonProcessor:
    """Highly optimized skeletonization and post-processing processor (supports 2D/3D input)"""

    def __init__(self, params: dict, im_info):
        self.im_info = im_info
        self.params = params
        self.raw_data = self.im_info.get_memmap('processed')
        self.is_3d = self.raw_data.ndim == 3

        # New: Hyperparameter for whether to compute labeled skeleton
        self.compute_labeled_skeleton = params.get('compute_labeled_skeleton', False)

        # Predefine neighborhood structures for optimization
        self._setup_neighborhoods()

    def _setup_neighborhoods(self):
        """Precompute neighborhood index templates"""
        if self.is_3d:
            # 3D 26-neighborhood (excluding center point)
            self.neighbor_offsets = list(product([-1, 0, 1], repeat=3))
            self.neighbor_offsets.remove((0, 0, 0))
            self.connectivity = 3
        else:
            # 2D 8-neighborhood (excluding center point)
            self.neighbor_offsets = list(product([-1, 0, 1], repeat=2))
            self.neighbor_offsets.remove((0, 0))
            self.connectivity = 2

    def skeletonize(self, binary: np.ndarray, orig_data: np.ndarray) -> tuple:
        """Optimized skeletonization processing"""
        print(f"Non-zero pixels before skeletonization: {np.count_nonzero(binary)}")

        # 1. Skeletonization
        if self.is_3d:
            skeleton = morphology.skeletonize_3d(binary)
        else:
            skeleton = morphology.skeletonize(binary)
        print(f"Non-zero pixels after skeletonization: {np.count_nonzero(skeleton)}")

        binary_skeleton = skeleton > 0

        # Decide whether to compute labeled skeleton based on hyperparameter
        if self.compute_labeled_skeleton:
            # Label propagation (compute only when needed)
            relabeled_binary = measure.label(binary, connectivity=self.connectivity)
            relabeled_skeleton = relabeled_binary * binary_skeleton

            # Optimize multi-label pixel removal using generic_filter
            relabeled_skeleton = self._remove_connected_label_pixels_generic_filter(relabeled_skeleton)

            # Optimized missing label addition
            relabeled_skeleton = self._add_missing_skeleton_labels_optimized(
                relabeled_skeleton, relabeled_binary, orig_data
            )
        else:
            # When labeled skeleton is not needed, set to empty array to save memory
            relabeled_binary = np.array([])
            relabeled_skeleton = np.array([])

        # Optimized pixel classification
        pixel_class = self._get_pixel_class_fast(binary_skeleton)

        print(f"Skeletonization post-processing complete")

        if self.compute_labeled_skeleton:
            return (binary_skeleton, pixel_class, relabeled_binary, relabeled_skeleton)
        else:
            return (binary_skeleton, pixel_class)

    def post_process(self, skeleton_data: tuple, orig_data: np.ndarray) -> tuple:
        """Optimized post-processing - entirely based on binary skeleton"""
        if self.compute_labeled_skeleton:
            binary_skeleton, relabeled_binary, relabeled_skeleton, pixel_class = skeleton_data
        else:
            binary_skeleton, pixel_class = skeleton_data

        # Quality control parameter adjustment - use binary skeleton
        qc_params = self._quality_control_fast(binary_skeleton, orig_data)

        # Merge parameters
        process_params = {**self.params, **qc_params}

        # Optimized post-processing - directly use binary skeleton
        processed_binary, pixel_class = self._postprocessing_binary_optimized(
            binary_skeleton, pixel_class, orig_data, process_params
        )

        # Determine return value based on whether labeled skeleton is needed
        if self.compute_labeled_skeleton and relabeled_skeleton.size > 0:
            # If labeled skeleton is needed, apply post-processing results to the labeled skeleton
            processed_labeled = self._apply_binary_to_labeled(
                processed_binary, relabeled_skeleton
            )
            return (processed_binary, processed_labeled, pixel_class)
        else:
            # If labeled skeleton is not needed, directly return binary results
            return (processed_binary, processed_binary.astype(np.uint32), pixel_class)

    def _apply_binary_to_labeled(self, processed_binary, original_labeled):
        """Apply binary post-processing results to the labeled skeleton"""
        # Create a new labeled skeleton, keeping only the post-processed skeleton points
        result = np.zeros_like(original_labeled)
        result[processed_binary] = original_labeled[processed_binary]
        return result

    def _postprocessing_binary_optimized(self, binary_skeleton: np.ndarray, pixel_class: np.ndarray,
                                         orig_data: np.ndarray, params: dict) -> tuple:
        """Optimized post-processing method based on binary skeleton"""
        min_object_size = params.get('min_object_size', 10)

        # Remove small objects
        skeleton_cleaned = morphology.remove_small_objects(
            binary_skeleton,
            min_size=min_object_size,
            connectivity=self.connectivity
        )

        processed_binary = skeleton_cleaned

        # Pruning parameters
        max_iterations = params.get('prune_iterations', 5)
        min_pixel_change = params.get('min_pixel_change', 10)
        min_branch_length = params.get('min_branch_length', 5)

        if not params.get('prune_skel', True):
            return processed_binary, self._get_pixel_class_fast(processed_binary)

        unchanged_iterations = 0
        prev_skeleton = None

        for iteration in range(max_iterations):
            current_binary = processed_binary
            if prev_skeleton is None:
                prev_skeleton = current_binary.copy()

            # Pruning method optimized with convolution
            processed_binary = self._prune_binary_with_convolution_endpoints(
                processed_binary, pixel_class, min_branch_length
            )

            # Update classification
            new_pixel_class = self._get_pixel_class_fast(processed_binary)

            # Calculate changes
            changed_pixels = np.sum(processed_binary != prev_skeleton)

            if changed_pixels <= min_pixel_change:
                unchanged_iterations += 1
                if unchanged_iterations >= 2:
                    print(f"Fast convergence: {iteration + 1} iterations, changed {changed_pixels} pixels")
                    break
            else:
                unchanged_iterations = 0

            prev_skeleton = processed_binary.copy()
            pixel_class = new_pixel_class

            if iteration < 3:
                print(f"Iteration {iteration + 1}: {np.sum(processed_binary)} pixels, changed {changed_pixels} pixels")

        return processed_binary, pixel_class

    def _prune_binary_with_convolution_endpoints(self, binary_skeleton: np.ndarray,
                                                 pixel_class: np.ndarray, min_length: int) -> np.ndarray:
        """Optimized pruning method based on binary skeleton"""
        if not np.any(binary_skeleton):
            return binary_skeleton

        # 1. Identify branch points
        branch_points = pixel_class >= 4
        branch_mask = np.zeros_like(binary_skeleton, dtype=bool)
        branch_mask[branch_points] = True

        # 2. Temporarily remove branch points
        temp_skel = binary_skeleton.copy().astype('uint8')
        temp_skel[branch_points] = 0

        # 3. Precompute all endpoints
        if self.is_3d:
            weights = np.ones((3, 3, 3))
        else:
            weights = np.ones((3, 3))

        # Calculate neighborhood sum
        temp_skel_sum = convolve(temp_skel, weights, mode='constant', cval=0)
        # Endpoints: neighborhood sum = 2 (self 1 + 1 neighbor)
        all_endpoints_mask = (temp_skel_sum == 2) & (temp_skel > 0)

        # 4. Label connected components
        labeled = measure.label(temp_skel, connectivity=self.connectivity)
        regions = measure.regionprops(labeled)

        # 5. Precompute connected components of branch points
        branch_labels = measure.label(branch_mask, connectivity=self.connectivity) if np.any(branch_mask) else None

        # 6. Create deletion mask
        delete_mask = np.zeros_like(binary_skeleton, dtype=bool)

        for region in regions:
            coords = region.coords
            num_pixels = len(coords)

            is_critical = False

            # Case 1: Single-pixel branch
            if num_pixels == 1:
                coord = tuple(coords)
                connected_regions = self._get_connected_branch_regions(coord, branch_labels, binary_skeleton.shape)
                is_critical = len(connected_regions) >= 2

            # Case 2: Multi-pixel branch
            else:
                # Quickly find endpoints using precomputed endpoint mask
                region_mask = (labeled == region.label)
                endpoints_mask = all_endpoints_mask & region_mask
                endpoints_coords = np.argwhere(endpoints_mask)
                endpoints = [tuple(coord) for coord in endpoints_coords]

                if len(endpoints) >= 2:
                    endpoint_regions = []
                    for endpoint in endpoints[:2]:  # Only check the first two endpoints
                        connected_regions = self._get_connected_branch_regions(endpoint, branch_labels,
                                                                               binary_skeleton.shape)
                        endpoint_regions.append(connected_regions)

                    # Critical connection judgment
                    is_critical = (endpoint_regions and endpoint_regions and
                                   not endpoint_regions.intersection(endpoint_regions))

            # Delete short non-critical branches
            if num_pixels < min_length and not is_critical:
                delete_mask[tuple(coords.T)] = True

        # 7. Apply deletion and restore branch points
        result = binary_skeleton.copy()
        result[delete_mask] = False
        result[branch_points] = True  # Restore branch points

        return result

    def _get_pixel_class_fast(self, skel):
        """Optimized pixel classification - fix background pixel errors"""
        skel_mask = (skel > 0).astype('uint8')

        if self.is_3d:
            weights = np.ones((3, 3, 3), dtype=np.uint8)
        else:
            weights = np.ones((3, 3), dtype=np.uint8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Calculate neighborhood sum (including self)
            skel_mask_sum = convolve(skel_mask, weights, mode='constant', cval=0)

        # Apply classification only to skeleton pixels, keep background at 0
        # Note: Here we only multiply by skel_mask, do not process the background
        skel_mask_sum = skel_mask_sum * skel_mask

        # Unify branch points greater than 4 as 4, but do not change the background value 0
        # The classification result is:
        # 0: Background
        # 1: Isolated point (very rare)
        # 2: Endpoint
        # 3: Connection point
        # 4: Branch point
        skel_mask_sum[skel_mask_sum > 4] = 4

        return skel_mask_sum

    def _remove_connected_label_pixels_generic_filter(self, skel_labels):
        """Multi-label pixel removal optimized using generic_filter"""

        def check_multiple_labels(neighborhood):
            """Check if there are multiple different labels in the neighborhood"""
            unique_labels = np.unique(neighborhood[neighborhood > 0])
            return len(unique_labels) > 1

        # Set neighborhood size
        ndim = skel_labels.ndim
        size = (3,) * ndim

        # Perform vectorized check using generic_filter
        multi_label_mask = generic_filter(
            skel_labels,
            check_multiple_labels,
            size=size,
            mode='constant',
            cval=0
        ).astype(bool)

        # Apply deletion only on skeleton points
        delete_mask = multi_label_mask & (skel_labels > 0)

        result = skel_labels.copy()
        result[delete_mask] = 0

        return result

    def _add_missing_skeleton_labels_optimized(self, skel_labels, label_frame, orig_data):
        """Highly optimized addition of missing labels"""
        unique_labels = np.unique(label_frame)
        unique_labels = unique_labels[unique_labels > 0]

        result = skel_labels.copy()

        # Ensure correct data type is used
        orig_data = orig_data.astype(np.float32)

        for label in unique_labels:
            if np.any(result == label):
                continue

            mask = label_frame == label
            if not np.any(mask):
                continue

            # Find the maximum value directly within the mask area to avoid creating a full temporary array
            masked_intensity = orig_data[mask]
            if masked_intensity.size == 0:
                continue

            max_val = np.max(masked_intensity)

            # Use np.where to find global maximum coordinates
            max_coords = np.where(orig_data == max_val)
            if not max_coords or len(max_coords) == 0:
                continue

            # Convert to coordinate list
            max_points = np.column_stack(max_coords)

            # Find the first point within the mask
            for point in max_points:
                point_tuple = tuple(point)
                if mask[point_tuple]:
                    result[point_tuple] = label
                    break

        return result

    def _get_connected_branch_regions(self, point, branch_labels, shape):
        """Get branch regions connected to a point"""
        connected_regions = set()

        if branch_labels is None:
            return connected_regions

        for offset in self.neighbor_offsets:
            neighbor_coord = tuple(point[i] + offset[i] for i in range(len(point)))

            # Boundary check
            if all(0 <= neighbor_coord[i] < shape[i] for i in range(len(point))):
                region_id = branch_labels[neighbor_coord]
                if region_id > 0:
                    connected_regions.add(region_id)

        return connected_regions

    def _quality_control_fast(self, processed, original):
        """Fast quality control"""
        qc_params = {
            'min_branch_length': self.params.get('min_branch_length', 5),
            'min_object_size': self.params.get('min_object_size', 10)
        }

        try:
            signal_mask = processed > 0
            if not np.any(signal_mask):
                return qc_params

            background_mask = ~signal_mask
            if not np.any(background_mask):
                return qc_params

            signal_mean = np.mean(original[signal_mask])
            background_std = np.std(original[background_mask])
            snr = signal_mean / (background_std + 1e-6)

            if snr < self.params.get('min_snr', 3):
                qc_params['min_branch_length'] = max(2, int(qc_params['min_branch_length'] * 1.2))
                qc_params['min_object_size'] = max(5, int(qc_params['min_object_size'] * 1.2))

        except Exception:
            pass

        return qc_params