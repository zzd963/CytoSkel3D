import os
import numpy as np
from skimage import io, filters, morphology, exposure, feature, util, img_as_ubyte, img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter, zoom, binary_closing, distance_transform_edt, convolve
import matplotlib
from datetime import datetime
from scipy.ndimage import median_filter
import tifffile
from skimage.transform import rotate
from skimage import measure
from scipy.spatial import cKDTree
from skan import summarize, Skeleton

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CytoSkel3D.information.Img_Information import Img_Information
from CytoSkel3D.preprocess.BinaryProcessor import BinaryProcessor
from CytoSkel3D.preprocess.RidgeEnhancer import RidgeEnhancer
from CytoSkel3D.preprocess.SkeletonProcessor import SkeletonProcessor


class HybridCytoskeletonProcessor:
    """Hybrid Strategy Cytoskeleton Processor (supports 2D/3D input)"""

    def __init__(self, im_info, params: dict, smoothed_image: np.ndarray = None,
                 ridge_image: np.ndarray = None, binary_image: np.ndarray = None):
        self.im_info = im_info
        self.params = params

        # Store provided preprocessed images
        self.provided_smoothed = smoothed_image
        self.provided_ridge = ridge_image
        self.provided_binary = binary_image

        # Initialize data attributes
        self.raw_data = im_info.get_memmap('processed')  # Raw data, read-only
        self.is_3d = self.raw_data.ndim == 3  # Automatically detect dimensions

        self.min_radius_px = params.get('min_radius_px', 1)  # Minimum radius of tubular structure (pixels)
        self.max_radius_px = params.get('max_radius_px', 10)  # Maximum radius of tubular structure (pixels)

        self.process_data = self.raw_data.copy()

        # Initialize processing parameters
        self.z_scale_factor = self._calculate_z_scale()
        self.apply_anisotropy = params.get('apply_anisotropic_scaling', False) and self.is_3d

        # Validate consistency of provided image dimensions
        self._validate_provided_images()

        # Initialize processing pipeline (dynamically adjusted based on provided images)
        self.step_order = self._get_adjusted_steps()
        self._validate_params()

        # Initialize processors
        self.skeleton_processor = SkeletonProcessor(self.params, self.im_info)
        self.binary_processor = BinaryProcessor(self.params, self.im_info)
        self.visualizing_method = False

    def _validate_provided_images(self):
        """Validate whether the dimensions of the provided preprocessed images match the raw data"""
        provided_images = [
            (self.provided_smoothed, "smoothed_image"),
            (self.provided_ridge, "ridge_image"),
            (self.provided_binary, "binary_image")
        ]

        for image, name in provided_images:
            if image is not None:
                # Check dimensional consistency
                if image.ndim != self.raw_data.ndim:
                    raise ValueError(f"The dimension of the provided {name} ({image.ndim}D) does not match the raw data ({self.raw_data.ndim}D)")

                # Check spatial size consistency (ignoring potential channel dimensions)
                if image.shape != self.raw_data.shape:
                    print(f"Warning: The size of {name} {image.shape} does not match the raw data size {self.raw_data.shape}")
                    # Resizing could be done here, but for safety, we just issue a warning

    def _get_adjusted_steps(self):
        """Dynamically adjust processing steps - skip corresponding steps based on provided preprocessed images and parameter configuration"""
        base_steps = [
            'smooth',
            'ridge',
            'contrast',
            'binary',
            'skeleton',
            'post_processing'
        ]

        # Get the step order configured in parameters
        configured_steps = self.params.get('preprocess_method', base_steps.copy())

        # If no preprocessed images are provided, return the configured steps directly
        if (self.provided_smoothed is None and
                self.provided_ridge is None and
                self.provided_binary is None):
            return configured_steps

        # Find steps to skip
        steps_to_remove = set()

        # Case 1: If a binarized image is provided, skip all steps before binarization (but do not skip binary itself)
        if self.provided_binary is not None:
            # Find the position of 'binary' in configured steps
            if 'binary' in configured_steps:
                binary_index = configured_steps.index('binary')
                # Remove all steps before 'binary' (excluding 'binary' itself)
                steps_to_remove.update(configured_steps[:binary_index])
                print(f"Detected provided binarized image, skipping steps: {configured_steps[:binary_index]}")

        # Case 2: If a ridge-enhanced image is provided, skip all steps before ridge enhancement (but do not skip ridge itself)
        elif self.provided_ridge is not None:
            if 'ridge' in configured_steps:
                ridge_index = configured_steps.index('ridge')
                # Remove all steps before 'ridge' (excluding 'ridge' itself)
                steps_to_remove.update(configured_steps[:ridge_index])
                print(f"Detected provided ridge-enhanced image, skipping steps: {configured_steps[:ridge_index]}")

        # Case 3: If a smoothed image is provided
        elif self.provided_smoothed is not None:
            if 'smooth' in configured_steps:
                # steps_to_remove.add('smooth')
                print("Detected provided smoothed image")

        # Build adjusted step list
        adjusted_steps = [step for step in configured_steps if step not in steps_to_remove]

        # Ensure essential steps exist (especially skeleton and post-processing)
        essential_steps = ['skeleton', 'post_processing']
        for essential_step in essential_steps:
            if (essential_step not in adjusted_steps and
                    essential_step in configured_steps):
                adjusted_steps.append(essential_step)

        return adjusted_steps

    # region Core Parameter Processing Logic
    def _calculate_z_scale(self):
        """Z-axis scaling factor calculation supporting adaptive/fixed modes"""
        if not self.is_3d:
            return 1.0

        # Prioritize checking fixed mode
        if self.params.get('z_scale_mode') == 'fixed':
            return self.params.get('fixed_z_scale', 1.0)

        # Dynamically calculate for adaptive mode
        try:
            z_pixel_size = self.im_info.pixel_size.get('Z', 1.0)
            xy_pixel_size = np.mean([
                self.im_info.pixel_size.get('X', 1.0),
                self.im_info.pixel_size.get('Y', 1.0)
            ])
            if z_pixel_size <= 0 or xy_pixel_size <= 0:
                raise ValueError("Pixel size must be greater than 0")
            return max(1.0, z_pixel_size / xy_pixel_size)
        except Exception as e:
            print(f"Anisotropy calculation failed: {str(e)}, using default value 1.0")
            return 1.0

    def _validate_params(self):
        """Enhanced parameter validation (dynamically adjust 3D parameters)"""
        required = [
            'gaussian_sigma',
            'ridge_method',
            'binary_method',
            'z_scale_mode',
            'apply_anisotropic_scaling',
            'min_branch_length'
        ]

        # fixed mode must provide fixed_z_scale
        if self.params.get('z_scale_mode') == 'fixed':
            required.append('fixed_z_scale')

        # Only 3D requires tubular length parameters
        if self.is_3d:
            required.append('min_object_size')

        missing = [p for p in required if p not in self.params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    # region Main Processing Pipeline
    def process_pipeline(self, visualize: bool = False, save_intermediates: bool = False,
                         visualize_method: str = None) -> tuple:
        """Execute hybrid processing pipeline, integrating preprocessed image support"""
        try:
            current_data = self.process_data.copy()
            # Backup raw data to handle upsampling scenarios
            self.orig_data_for_skeleton = current_data.copy()

            # Anisotropic preprocessing (if needed)
            if self.apply_anisotropy:
                current_data = self._anisotropic_scaling(current_data, mode='upsample')
                # Upsample raw data as well
                self.orig_data_for_skeleton = self._anisotropic_scaling(
                    self.orig_data_for_skeleton, mode='upsample')

                # Upsample provided preprocessed images (if they exist)
                if self.provided_smoothed is not None:
                    self.provided_smoothed = self._anisotropic_scaling(
                        self.provided_smoothed, mode='upsample')
                if self.provided_ridge is not None:
                    self.provided_ridge = self._anisotropic_scaling(
                        self.provided_ridge, mode='upsample')
                if self.provided_binary is not None:
                    self.provided_binary = self._anisotropic_scaling(
                        self.provided_binary, mode='upsample')

            intermediates = self._execute_pipeline_steps(current_data)

            # Anisotropic restoration (if needed)
            if self.apply_anisotropy:
                intermediates = self._downscale_intermediates(intermediates)

            if save_intermediates:
                self._save_results(intermediates)
            if visualize:
                self._visualize_steps(intermediates)

            # Call method for visualization
            self._visualize_method_condition(visualize_method)

            return intermediates

        except Exception as e:
            self._log_error(f"Processing failed: {str(e)}")
            raise

    def _execute_pipeline_steps(self, input_data: np.ndarray) -> dict:
        """Execute each step of the processing pipeline (supports preprocessed image integration)"""
        current_data = input_data.copy()
        intermediates = {}

        # Step-by-step processing
        for step in self.step_order:
            # Check if preprocessed results are provided for this step
            provided_result = self._get_provided_step_result(step)
            if provided_result is not None:
                # Use provided preprocessed results
                current_data = provided_result
                print(f"Using provided {step} image, skipping processing")
            else:
                # Execute processing step normally
                processor = self._get_step_processor(step)
                current_data = processor(current_data)

            intermediates[step] = current_data

        return intermediates

    def _get_provided_step_result(self, step: str) -> np.ndarray or None:
        """Get the provided preprocessed image corresponding to the step"""
        step_mapping = {
            'smooth': self.provided_smoothed,
            'ridge': self.provided_ridge,
            'binary': self.provided_binary
        }
        return step_mapping.get(step, None)

    def _get_step_processor(self, step: str) -> callable:
        """Get the processor function corresponding to the processing step"""
        processor_mapping = {
            'smooth': self._denoise,
            'ridge': self._ridge,
            'contrast': self._contrast,
            'binary': self._binary,
            'skeleton': self._skeleton,
            'post_processing': self._post_processing
        }
        return processor_mapping.get(step, self._default_processor)

    def _downscale_intermediates(self, intermediates: dict) -> dict:
        """Downsample intermediate results of 3D processing (Z-axis restoration)"""
        if not self.is_3d or not self.apply_anisotropy:
            return intermediates

        for step in intermediates:
            if step == 'skeleton' or step == 'post_processing':
                # Handle special steps that return tuples
                processed_data = []
                for data in intermediates[step]:
                    downscaled = self._anisotropic_scaling(data, mode='downsample')
                    processed_data.append(downscaled)
                intermediates[step] = tuple(processed_data)
            else:
                intermediates[step] = self._anisotropic_scaling(
                    intermediates[step], mode='downsample'
                )
        return intermediates

    def _default_processor(self, image: np.ndarray) -> np.ndarray:
        """Default processor: directly return input image"""
        return image

    # endregion

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Select denoising method based on parameters (skip if preprocessed image is provided)"""
        # If a smoothed image is provided but processing is needed here, it's an anomaly
        if self.provided_smoothed is not None:
            print("Warning: Smoothed image provided but smoothing step is still executing, using provided image")
            return self.provided_smoothed

        # First execute background zeroing (if enabled)
        if self.params.get('pre_threshold_zeroing', False):
            threshold = self.params.get('zeroing_threshold', 0.1)
            mean_val = np.mean(image)
            mask = image < (threshold * mean_val)
            image[mask] = 0

        method = self.params.get('smooth_method', 'gaussian')

        # Handle "no_denoise" case - directly return raw image
        if method == 'no_denoise':
            return image

        # Gaussian filtering (isotropic processing)
        if method == 'gaussian':
            sigma = self.params.get('gaussian_sigma', 1.0)
            if self.is_3d:
                sigma_vec = (sigma, sigma, sigma)
            else:
                sigma_vec = (sigma, sigma) if image.ndim == 2 else (0, sigma, sigma)
            return gaussian_filter(image, sigma=sigma_vec)

        # Median filtering (isotropic processing)
        elif method == 'median':
            size = self.params.get('median_size', 3)
            return median_filter(image, size=size)

        # Top-hat filtering (preserves edges of tubular structures)
        elif method == 'tophat':
            radius = self.params.get('tophat_radius', 5)
            if self.is_3d:
                # 3D structuring element: ball
                selem = morphology.ball(radius)
            else:
                # 2D structuring element: disk
                selem = morphology.disk(radius)

            # Apply white top-hat transform (original image minus opening result)
            return morphology.white_tophat(image, selem)

        # Default to returning original image
        else:
            return image

    def _ridge(self, image: np.ndarray) -> np.ndarray:
        """Process image using ridge enhancer (skip if preprocessed image is provided)"""
        if self.provided_ridge is not None:
            print("Warning: Ridge-enhanced image provided but ridge enhancement step is still executing, using provided image")
            return self.provided_ridge

        self.ridge_enhancer = RidgeEnhancer(self.params, self.im_info)
        return self.ridge_enhancer.process(image)

    def _contrast(self, image: np.ndarray) -> np.ndarray:
        """Unified contrast adjustment method"""
        # Calculate global percentiles
        p_low, p_high = np.percentile(image, self.params.get('contrast_range', (2, 98)))

        # Unified processing for 2D and 3D
        return exposure.rescale_intensity(image, in_range=(p_low, p_high))

    def _binary(self, image: np.ndarray) -> np.ndarray:
        """Use independent binarization processor (skip if preprocessed image is provided)"""
        if self.provided_binary is not None:
            print("Warning: Binarized image provided but binarization step is still executing, using provided image")
            return self.provided_binary

        return self.binary_processor.process(image)

    def _skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Execute skeletonization using skeleton processor"""
        if self.visualizing_method == True:
            orig_data = self.process_data
        else:
            orig_data = self.orig_data_for_skeleton
        return self.skeleton_processor.skeletonize(binary, orig_data)

    def _post_processing(self, skeleton_data: tuple) -> np.ndarray:
        """Execute post-processing using skeleton processor"""
        return self.skeleton_processor.post_process(skeleton_data, self.process_data)

    def _anisotropic_scaling(self, data: np.ndarray, mode: str) -> np.ndarray:
        """Anisotropic scaling (upsampling/downsampling)"""
        if not self.is_3d or not self.apply_anisotropy:
            return data

        try:
            if mode == 'upsample':
                factors = (self.z_scale_factor, 1, 1)
                return zoom(data, factors, order=1)
            elif mode == 'downsample':
                factors = (1 / self.z_scale_factor, 1, 1)
                return zoom(data, factors, order=0)
            else:
                raise ValueError(f"Invalid scaling mode: {mode}")
        except Exception as e:
            self._log_error(f"Anisotropic scaling failed: {str(e)}")
            return data

    # endregion

    # region Result Saving and Visualization
    def _save_results(self, results: dict):
        """Fix intermediate result saving issue"""
        path_mapping = {
            'smooth': 'im_pre_smoothed',
            'ridge': 'im_pre_ridge',
            'contrast': 'im_pre_contrast',
            'binary': 'im_pre_binary',
            'skeleton': ['im_pre_skeleton', 'im_pre_skeleton_pixel_class', 'im_pre_binary_relabeled', 'im_pre_skeleton_relabeled'],
            'post_processing': ['im_pre_cleaned_skeleton', 'im_pre_cleaned_skeleton_relabeled',
                                'im_pre_cleaned_skeleton_pixel_class']
        }
        if self.skeleton_processor.compute_labeled_skeleton == False:
            path_mapping['skeleton'] = ['im_pre_skeleton', 'im_pre_skeleton_pixel_class']

        # Ensure output directory exists
        output_dir = self.im_info.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for step_name, data in results.items():
            # Skip steps not meant to be saved
            if step_name not in self.params['output_method'] and step_name not in path_mapping:
                continue

            try:
                # Special handling for skeleton and post-processing steps (they return tuples)
                if step_name == 'skeleton' or step_name == 'post_processing':
                    # Get the output key list for this step
                    output_keys = path_mapping[step_name]

                    # Iterate through each element in the tuple
                    for idx, output_data in enumerate(data):
                        if idx < len(output_keys):
                            key = output_keys[idx]
                            save_path = self.im_info.pipeline_paths.get(
                                key,
                                os.path.join(output_dir, f'{key}.tif')
                            )
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            print(f"Saving {step_name} {key} to: {save_path}")

                            # Process 2D data dimensions
                            if not self.is_3d and isinstance(output_data, np.ndarray) and output_data.ndim == 3 and \
                                    output_data.shape == 1:
                                output_data = output_data

                            # Save based on data type
                            if key in ['im_pre_skeleton', 'im_pre_cleaned_skeleton']:
                                # Save binary skeleton as binary image
                                io.imsave(save_path, img_as_ubyte(output_data))
                            else:
                                # Otherwise, save as 8-bit or 16-bit
                                if output_data.dtype == bool:
                                    io.imsave(save_path, img_as_ubyte(output_data))
                                else:
                                    io.imsave(save_path, output_data.astype(np.uint16))
                else:
                    # Other steps save a single image directly
                    pipeline_key = path_mapping.get(step_name, step_name)
                    save_path = self.im_info.pipeline_paths.get(
                        pipeline_key,
                        os.path.join(output_dir, f'{pipeline_key}.tif')
                    )
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print(f"Saving {step_name} to: {save_path}")

                    # Process 2D data dimensions
                    if not self.is_3d and isinstance(data, np.ndarray) and data.ndim == 3 and data.shape == 1:
                        data = data

                    # Fix special handling for skeleton data
                    if pipeline_key in ['im_pre_binary']:
                        # Save binary skeleton data
                        io.imsave(save_path, img_as_ubyte(data))
                    elif pipeline_key in ['im_pre_ridge']:
                        data = exposure.rescale_intensity(data, out_range=(0, 1))
                        io.imsave(save_path, img_as_ubyte(data))
                    else:
                        # Save as TIFF
                        if data.dtype == bool:
                            io.imsave(save_path, img_as_ubyte(data))
                        else:
                            io.imsave(save_path, data.astype(np.uint16))

            except Exception as e:
                error_msg = f"Error saving {step_name} step results: {str(e)}"
                print(error_msg)
                self._log_error(error_msg)

    def _visualize_method_condition(self, visualize_method: str or list):
        """Execute corresponding visualization based on passed method type"""
        if visualize_method is None:
            return

        # Define supported visualization method types and their configurations
        method_configs = {
            'ridge': {
                'method_type': 'ridge',
                'method_list': list(self.params['ridge_params'].keys()),
                'param_key': 'ridge_method',
                'filename_suffix': 'ridge'
            },
            'denoise': {
                'method_type': 'denoise',
                'method_list': ['no_denoise', 'gaussian', 'median', 'tophat'],
                'param_key': 'smooth_method',
                'filename_suffix': 'denoise'
            },
            'binary': {
                'method_type': 'binary',
                'method_list': ['otsu', 'multiotsu', 'minimum_cross_entropy'],
                'param_key': 'binary_method',
                'filename_suffix': 'binary'
            }
        }

        # Handle 'all' case - visualize all methods
        if visualize_method == 'all':
            for config in method_configs.values():
                self._visualize_all_methods(**config)
            return

        # Handle single method type
        if isinstance(visualize_method, str):
            if visualize_method in method_configs:
                self._visualize_all_methods(**method_configs[visualize_method])
            else:
                print(f"Warning: Unknown visualization method type '{visualize_method}', ignored")
            return

        # Handle list of methods
        if isinstance(visualize_method, list):
            for method in visualize_method:
                if method in method_configs:
                    self._visualize_all_methods(**method_configs[method])
                else:
                    print(f"Warning: Unknown visualization method type '{method}' in list, ignored")
            return

        print(f"Warning: Invalid visualization method parameter type '{type(visualize_method)}', ignored")

    def _visualize_all_methods(self, method_type: str, method_list: list, param_key: str, filename_suffix: str):
        """General method: Visualize all method results of a specified type"""
        # Save original method settings
        self.visualizing_method = True
        original_method = self.params[param_key]
        all_results = []

        # Save original processing data
        original_process_data = self.process_data.copy()

        for method in method_list:
            print(f"Running {method_type} method: {method}")
            self.params[param_key] = method

            # Reset processing data to original state
            self.process_data = original_process_data.copy()

            # Use core processing function
            intermediates = self._execute_pipeline_steps(self.process_data)
            all_results.append(intermediates)

        # Restore original parameters and processing data
        self.params[param_key] = original_method
        self.process_data = original_process_data

        # Draw results for all methods
        save_path = os.path.join(
            self.im_info.graph_dir,
            f'processing_summary_all_{filename_suffix}.png'
        )
        self._visualize_all_methods_results(method_list, all_results, save_path, method_type)

        self.visualizing_method = False

    def _visualize_all_methods_results(self, method_list: list, all_results: list, save_path: str, method_type: str):
        """Draw comparison chart of results for all methods (Rows: Methods, Columns: Processing Steps)"""
        # Calculate chart size
        n_rows = len(method_list)
        n_cols = len(self.step_order)
        figsize = (min(20, 5 * n_cols), min(20, 5 * n_rows))

        # Create chart and axes
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False
        )

        # Generate list of step titles
        step_titles = {
            'smooth': "Denoised",  # Preprocessed
            'ridge': "Ridge Enhanced",
            'contrast': "Contrast Adjusted",
            'binary': "Binarized",
            'skeleton': "Skeletonised",
            'post_processing': 'Cleaned'
        }

        # Draw each step for each method
        for i, method in enumerate(method_list):
            intermediates = all_results[i]
            for j, step in enumerate(self.step_order):
                ax = axes[i, j]

                if step in intermediates:
                    data = intermediates[step]

                    # Special handling for skeleton step
                    if step == 'skeleton':
                        # Skeleton step uses binary skeleton (first element of tuple)
                        display_data = data

                    # Special handling for post-processing step
                    elif step == 'post_processing':
                        # Post-processing step uses binary result (first element of tuple)
                        display_data = data

                    # Other steps use data directly
                    else:
                        display_data = data

                    # For 3D data, use maximum projection
                    if display_data.ndim == 3:
                        display_img = np.max(display_data, axis=0)
                    else:
                        display_img = display_data

                    # Ensure data is within 0-1 range
                    if np.min(display_img) < 0 or np.max(display_img) > 1:
                        display_img = exposure.rescale_intensity(display_img, out_range=(0, 1))

                    ax.imshow(display_img, cmap='gray')

                # Set titles
                if i == 0:
                    ax.set_title(step_titles.get(step, step), fontsize=10)
                if j == 0:
                    ax.set_ylabel(method, fontsize=10)

                ax.axis('off')

        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"{method_type.capitalize()} method comparison chart saved to: {save_path}")

    def _visualize_steps(self, results: dict):
        """Dynamic visualization"""
        # Generate list of step titles
        step_titles = {
            'smooth': "Denoised",  # Preprocessed
            'ridge': "Ridge Enhanced",
            'contrast': "Contrast Adjusted",
            'binary': "Binarized",
            'skeleton': "Skeletonised",
            'post_processing': 'Cleaned'
        }

        # Get steps to display (steps in output_method and present in results)
        steps_to_display = [
            step for step in self.params['output_method']
            if step in results
        ]

        # Prepare visualization data
        volumes = []
        titles = []
        for step in steps_to_display:
            data = results[step]

            # Special handling for skeleton step - use binary skeleton
            if step == 'skeleton':
                if self.skeleton_processor.compute_labeled_skeleton == False:
                    binary_skeleton, pixel_class = data
                else:
                    binary_skeleton, pixel_class, relabeled_binary, relabeled_skeleton = data
                volumes.append(binary_skeleton)

            # Special handling for post-processing step - use post-processed binary skeleton
            elif step == 'post_processing':
                binary_result, _, _ = data
                volumes.append(binary_result)

            # Other steps use data directly
            else:
                volumes.append(data)

            titles.append(step_titles.get(step, step))

        # Generate visualization chart
        if len(volumes) > 0:
            fig = self._visualize_steps_figure(volumes, titles)
            fig.savefig(
                os.path.join(self.im_info.graph_dir, 'processing_summary.png'),
                dpi=300
            )
            plt.close(fig)

    def _visualize_steps_figure(self, volumes, titles):
        """Create visualization chart"""
        fig, axes = plt.subplots(1, len(volumes), figsize=(16, 8))
        for i, (vol, title) in enumerate(zip(volumes, titles)):
            ax = axes[i] if len(volumes) > 1 else axes

            if vol.ndim == 3:
                # Use maximum projection
                display_img = np.max(vol, axis=0)
            else:
                display_img = vol

            # Ensure data is within 0-1 range
            if np.min(display_img) < 0 or np.max(display_img) > 1:
                display_img = exposure.rescale_intensity(display_img, out_range=(0, 1))

            ax.imshow(display_img, cmap='gray')

            ax.set_title(title)
            ax.axis('off')

        return fig

    def _log_error(self, message: str):
        """Unified error logging"""
        log_path = os.path.join(self.im_info.output_dir, 'process_errors.log')
        with open(log_path, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")


# Updated parameter configuration (remains unchanged)
HYBRID_PARAMS = {
    'preprocess_method': [
        'smooth',
        'ridge',
        'binary',
        'skeleton',
        'post_processing'
    ],  # Optimized pipeline based on DRAGoN algorithm

    # Smoothing method parameters
    'smooth_method': 'gaussian',
    'gaussian_sigma': 1.0,
    'median_size': 3,
    'tophat_radius': 15,

    # Background zeroing parameters
    'pre_threshold_zeroing': True,
    'zeroing_threshold': 0.8,

    # Ridge enhancement filter selection parameters
    'ridge_method': 'Meijering',
    'ridge_params': {
        'Sato': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Frangi': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Frangi_Sato': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Meijering': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Hessian': {
            'sigma': 1
        },
        'multi_orient_skel': {
            'angles': [0, 30, 60, 90, 120, 150],
            'sigmas': np.linspace(1, 4, 5)
        }
    },

    # Tubular structure size parameters
    'min_radius_px': 1,
    'max_radius_px': 5,

    # Binarization parameters
    'binary_strategy': 'Global',
    'binary_method': 'otsu',
    'threshold_correction_factor': 1.0,
    'log_transform': False,
    'assign_middle_to_foreground': "foreground",
    'adaptive_block_size': 50,
    'global_limits': [0.7, 1.5],

    'min_noise_size': 15,
    'apply_morphology': True,
    'morph_radius': 1,

    # Post-processing optimization parameters
    'quality_control': True,
    'min_snr': 2.5,
    'min_object_size': 10,
    'min_branch_length': 3,
    'prune_iterations': 10,
    'min_pixel_change': 1,
    'prune_skel': True,

    'output_method': [
        'binary',
        'skeleton',
        'post_processing'
    ],

    # Anisotropic processing parameters
    'apply_anisotropic_scaling': False,
    'z_scale_mode': 'adaptive',
    'fixed_z_scale': 1.0,
    'voxel_size': (1, 1, 1),
}

if __name__ == "__main__":
    # Test cases - demonstrate new features
    test_dir =
    out_dir =

    all_paths = [f for f in os.listdir(test_dir) if f.endswith(('.tiff', '.tif'))]
    test_file = os.path.join(test_dir, all_paths)

    img_info = Img_Information(test_file, output_dir=out_dir)
    img_info.change_axes('ZYX')
    img_info.change_pixel_size('Z', 1)
    img_info.create_save_tiff()

    # Example 1: Normal processing (no preprocessed images)
    processor1 = HybridCytoskeletonProcessor(img_info, HYBRID_PARAMS)
    result1 = processor1.process_pipeline(visualize=True, save_intermediates=True)

    # Example 2: Using provided binarized image (skips all previous steps)
    # binary_img = ... # Binarized image obtained from other research groups
    # processor2 = HybridCytoskeletonProcessor(img_info, HYBRID_PARAMS, binary_image=binary_img)
    # result2 = processor2.process_pipeline(visualize=True, save_intermediates=True)

    print("Processing complete, results saved in:", out_dir)