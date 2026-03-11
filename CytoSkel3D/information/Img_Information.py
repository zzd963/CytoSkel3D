import os
import copy
import numpy as np
from tifffile import tifffile
from PIL import Image
import json
from datetime import datetime


class Img_Information:
    """
    File information processing class, supports multi-path generation
    """

    def __init__(self, filepath, output_dir=None, maskpath=None):
        # File path related attributes
        self.filepath = filepath
        self.input_dir = os.path.dirname(filepath)
        self.basename = os.path.basename(filepath)
        self.maskpath = maskpath

        self.filename_no_ext = os.path.splitext(self.basename)
        self.extension = os.path.splitext(filepath)
        self.output_dir = output_dir or os.path.join(self.input_dir, 'output')

        # Create directory structure
        self.necessities_dir = os.path.join(self.output_dir, 'necessities')
        self.screenshot_dir = os.path.join(self.output_dir, 'screenshots')
        self.graph_dir = os.path.join(self.output_dir, 'graphs')
        self.feature_dir = os.path.join(self.output_dir, 'features')
        os.makedirs(self.necessities_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

        # Metadata attributes
        self.metadata = {}
        self.metadata_type = None
        self.original_axes = None  # New: Save original axis order
        self.axes = None
        self.shape = None
        self.pixel_size = {'X': None, 'Y': None, 'Z': None, 'T': None}
        self.dtype = None
        self.good_dims = False
        self.good_axes = False
        self._voxel_volume = None  # No need to initialize, dynamically calculated by property

        # Processing parameters
        self.ch = 0
        self.t_start = 0
        self.t_end = None
        self.z_start = 0
        self.z_end = None

        # Initialize processing paths
        self.pipeline_paths = {}
        self.output_path = None
        self._create_output_paths()

        # Metadata loading process
        self.find_metadata()
        self.load_metadata()
        # Then validate mask (new sequence control)
        if self.maskpath:
            self._validate_mask()
        self._validate()

    def _create_output_paths(self):
        """Create all processing paths"""
        # Main output path
        self.pipeline_paths = {
            'processed': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_processed.tif"),
            'mask': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_mask.tif"),

            'im_pre_inverted': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_inverted.tif"),
            'im_pre_smoothed': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_smoothed.tif"),
            'im_pre_zeroing': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_zeroing.tif"),
            'im_pre_ridge': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_ridge.tif"),
            'im_pre_contrast': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_contrast.tif"),

            'im_pre_binary': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_binary.tif"),
            'im_pre_binary_relabeled': os.path.join(self.necessities_dir,
                                                    f"{self.filename_no_ext}_pre_binary_relabeled.tif"),

            'im_pre_skeleton': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_pre_skeleton.tif"),
            'im_pre_skeleton_relabeled': os.path.join(self.necessities_dir,
                                                      f"{self.filename_no_ext}_pre_skeleton_relabeled.tif"),
            'im_pre_skeleton_pixel_class': os.path.join(self.necessities_dir,
                                                        f"{self.filename_no_ext}_pre_skeleton_pixel_class.tif"),

            'im_pre_cleaned_skeleton': os.path.join(self.necessities_dir,
                                                    f"{self.filename_no_ext}_pre_cleaned_skeleton.tif"),
            'im_pre_cleaned_skeleton_relabeled': os.path.join(self.necessities_dir,
                                                              f"{self.filename_no_ext}_pre_cleaned_skeleton_relabeled.tif"),
            'im_pre_cleaned_skeleton_pixel_class': os.path.join(self.necessities_dir,
                                                                f"{self.filename_no_ext}_pre_cleaned_skeleton_pixel_class.tif"),

            'im_reconstruct_segments': os.path.join(self.necessities_dir,
                                                    f"{self.filename_no_ext}_reconstruct_segments.tif"),
            'im_reconstruct_segments_reconstructed': os.path.join(self.necessities_dir,
                                                                  f"{self.filename_no_ext}_reconstruct_segments_reconstructed.tif"),
            'im_reconstruct_skeleton': os.path.join(self.necessities_dir,
                                                    f"{self.filename_no_ext}_reconstruct_skeleton.tif"),

            'im_preprocessed': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_preprocessed.tif"),

            'generated_mask': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_generated_mask.tif"),
            'generated_object_mask': os.path.join(self.necessities_dir,
                                                  f"{self.filename_no_ext}_generated_object_mask.tif"),

            'im_instance_label': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_instance_label.tif"),
            'im_skel': os.path.join(self.necessities_dir, f"{self.filename_no_ext}_skeleton.tif"),
            'analysis_results': os.path.join(self.output_dir, f"{self.filename_no_ext}_results.csv")
        }
        self.output_path = self.pipeline_paths['processed']  # Set default main path

    def load_metadata(self):
        """Load dimension information"""
        self._get_basic_tiff_metadata()
        self._validate()

    def find_metadata(self):
        """Metadata discovery entry point"""
        if self.filepath.lower().endswith(('.tiff', '.tif')):
            self._find_tif_metadata()
        else:
            raise ValueError('Only TIFF format is supported')

    def _find_tif_metadata(self):
        """Extract TIFF metadata"""
        with tifffile.TiffFile(self.filepath) as tif:
            self.axes = tif.series.axes
            self.shape = tif.series.shape
            page = tif.pages
            self.metadata = {tag.name: tag.value for tag in page.tags.values()}
            self.metadata_type = 'basic_tiff'

    def _get_basic_tiff_metadata(self):
        """Parse pixel size information"""
        res_unit = self.metadata.get('ResolutionUnit')
        unit_scale = 1.0

        if res_unit == tifffile.TIFF.RESUNIT.CENTIMETER:
            unit_scale = 1e4  # Centimeter to micrometer conversion factor
        elif res_unit == tifffile.TIFF.RESUNIT.INCH:
            unit_scale = 25400  # Inch to micrometer conversion factor

        # Process X/Y pixel size
        for axis in ['X', 'Y']:
            if tag := self.metadata.get(f'{axis}Resolution'):
                if tag != 0:
                    # Calculate actual pixel size (unit: micrometer)
                    self.pixel_size[axis] = (tag / tag) * unit_scale

        # Process Z pixel size
        if 'ZResolution' in self.metadata and 'Z' in self.axes:
            self.pixel_size['Z'] = self.metadata['ZResolution']

    def change_pixel_size(self, dim, new_size):
        """
        Modify pixel size
        :param dim: Dimension to modify (X/Y/Z/T)
        :param new_size: New pixel size value (unit: micrometer)
        """
        valid_dims = {'X', 'Y', 'Z', 'T'}
        if dim not in valid_dims:
            print(f"Error: Invalid dimension {dim}, supported dimensions: {valid_dims}")
            return

        if not isinstance(new_size, (int, float)) or new_size <= 0:
            print("Error: Pixel size must be positive")
            return

        self.pixel_size[dim] = new_size
        self._validate()
        print(f"{dim}-axis pixel size updated to: {new_size} μm")

    @property
    def voxel_volume(self):
        """Dynamically calculate voxel volume (unit: cubic micrometer)"""
        try:
            return self.pixel_size['X'] * self.pixel_size['Y'] * self.pixel_size['Z']
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f"Cannot calculate voxel volume, please check if X/Y/Z pixel sizes are set correctly. "
                f"Current pixel sizes: X={self.pixel_size['X']}, Y={self.pixel_size['Y']}, Z={self.pixel_size['Z']}"
            ) from e

    def change_axes(self, new_axes):
        """
        Modify axis order
        :param new_axes: New axis order string, e.g., 'TZYX'
        """
        # Basic validation
        if len(new_axes) != len(self.axes):
            print(f"Error: Number of new axes ({len(new_axes)}) does not match original ({len(self.axes)})")
            return

        # Check for required axes
        if 'X' not in new_axes or 'Y' not in new_axes:
            print("Error: New axis order must contain X and Y axes")
            return

        # Check for duplicate axes
        if len(set(new_axes)) != len(new_axes):
            print("Error: Duplicate axes detected")
            return

        self.axes = new_axes
        self._validate()
        print(f"Axis order updated to: {new_axes}")

    def select_temporal_range(self, start=0, end=None):
        """
        Select temporal range
        :param start: Start time point
        :param end: End time point (None means the end)
        """
        if 'T' not in self.axes:
            print("Warning: Current data does not contain a time dimension")
            return

        max_time = self.shape[self.axes.index('T')] - 1

        # Parameter validation
        if start < 0 or start > max_time:
            print(f"Error: Start time point {start} is out of bounds [0-{max_time}]")
            return

        if end is not None and (end < start or end > max_time):
            print(f"Error: End time point {end} is invalid")
            return

        self.t_start = start
        self.t_end = end or max_time
        print(f"Temporal range set to: {start}-{self.t_end}")

    def select_channel(self, channel: int):
        """
        Select processing channel
        :param channel: Channel index (0-based)
        """
        # Validate existence of channel dimension
        if 'C' not in self.axes:
            print("Warning: Current data does not contain a channel dimension")
            return

        # Validate channel index
        c_index = self.axes.index('C')
        total_channels = self.shape[c_index]

        if channel < 0 or channel >= total_channels:
            raise IndexError(
                f"Invalid channel index {channel}, valid range: [0-{total_channels - 1}]"
            )

        self.ch = channel
        print(f"Selected channel {channel} (Total {total_channels} channels)")

    def _check_axes(self):
        """Axis order validation"""
        self.good_axes = (
                'X' in self.axes and
                'Y' in self.axes and
                len(set(self.axes)) == len(self.axes)
        )

    def _check_pixel_size(self):
        """Pixel size validation"""
        required_dims = set(self.axes) & {'X', 'Y', 'Z', 'T'}
        self.good_dims = all(self.pixel_size[dim] for dim in required_dims)

    def get_memmap(self, path_key: str):
        """Get memory map of specified processing stage"""
        if path_key not in self.pipeline_paths:
            raise KeyError(f"Invalid path key: {path_key}")
        return tifffile.memmap(self.pipeline_paths[path_key])

    def remove_intermediates(self):
        """Clean up intermediate files (keep analysis results)"""
        for key, path in self.pipeline_paths.items():
            if key != 'analysis_results' and os.path.exists(path):
                os.remove(path)

    def _validate(self):
        """Comprehensive validation"""
        self._check_axes()
        self._check_pixel_size()
        self._create_output_paths()  # Replaced with new path creation method

    # def _validate_mask(self):
    #     """Validate mask file"""
    #     if not os.path.exists(self.maskpath):
    #         raise FileNotFoundError(f"Mask file does not exist: {self.maskpath}")
    #
    #     # Validate mask and image size match
    #     with tifffile.TiffFile(self.maskpath) as tif:
    #         mask_shape = tif.series.shape
    #         if mask_shape != self.shape:
    #             raise ValueError(f"Mask shape {mask_shape} does not match image shape {self.shape}")

    def _validate_mask(self):
        """Simplified size validation, focusing on spatial dimensions"""
        if not os.path.exists(self.maskpath):
            raise FileNotFoundError(f"Mask file does not exist: {self.maskpath}")

        # Get Y/X size of the original image (last two dimensions)
        try:
            original_yx = self.shape[-2], self.shape[-1]  # Assume Y/X are the last two dimensions
        except IndexError:
            raise ValueError("Insufficient original image dimensions to get spatial size")

        # Get spatial size of the mask
        if self.maskpath.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(self.maskpath) as tif:
                mask_shape = tif.series.shape
                mask_yx = mask_shape[-2], mask_shape[-1]  # Take the last two dimensions
        else:
            with Image.open(self.maskpath) as img:
                # PIL size is (width, height), convert to (height, width)
                mask_yx = img.size[::-1]  # Reverse order to get (height, width)

        # Directly compare tuples
        if mask_yx != original_yx:
            raise ValueError(
                f"Mask spatial size {mask_yx} (H×W) does not match image size {original_yx}\n"
                f"Hint: When using multi-dimensional TIFF, ensure Y/X are the last two dimensions"
            )

    def create_save_tiff(self):
        # Initialize main image data
        if not os.path.exists(self.output_path):
            self.save_tiff()

    def save_tiff(self):
        """Save processed TIFF"""
        data = self._process_data()

        # If mask path is provided, set non-mask parts to zero
        if self.maskpath:
            # Read mask data
            if self.maskpath.lower().endswith(('.tif', '.tiff')):
                mask_data = tifffile.imread(self.maskpath)
            else:
                # Use PIL to process non-TIFF images
                with Image.open(self.maskpath) as img:
                    mask_data = np.array(img)

            # Ensure mask data has the same dimensions as processed data
            if mask_data.ndim != data.ndim:
                # If mask is 2D and data is 3D, expand mask to 3D
                if mask_data.ndim == 2 and data.ndim == 3:
                    mask_data = np.repeat(mask_data[np.newaxis, :, :], data.shape, axis=0)
                # If mask is 3D and data is 2D, take the first layer of the mask
                elif mask_data.ndim == 3 and data.ndim == 2:
                    mask_data = mask_data[0, :, :]
                else:
                    print(f"Warning: Mask dimensions ({mask_data.ndim}) do not match data dimensions ({data.ndim})")

            # Set non-mask parts to zero - using np.where is safer and more efficient
            data = np.where(mask_data > 0, data, 0)

            # Save mask file
            mask_save_path = self.pipeline_paths['mask']
            # Create serializable metadata
            serializable_metadata = {}
            for key, value in self.metadata.items():
                # Process bytes type: convert to hex string or ignore
                if isinstance(value, bytes):
                    try:
                        # Attempt UTF-8 decoding, if failed convert to hex representation
                        serializable_metadata[key] = value.decode('utf-8')
                    except UnicodeDecodeError:
                        # Binary data that cannot be decoded is converted to hex string
                        serializable_metadata[key] = value.hex()
                # Process numpy arrays
                elif isinstance(value, np.ndarray):
                    serializable_metadata[key] = value.tolist()
                # Process other serializable types
                elif isinstance(value, (str, int, float, bool, list, tuple, dict)):
                    serializable_metadata[key] = value
                # Ignore unserializable types
                else:
                    print(f"Warning: Ignored unserializable metadata key: {key} ({type(value)})")

            tifffile.imwrite(
                mask_save_path,
                mask_data,
                bigtiff=True,
                # metadata=serializable_metadata  # Copy original metadata
            )
            print(f"Mask saved to: {mask_save_path}")

        # # Create serializable metadata
        # serializable_metadata = {
        #     'axes': self.axes,
        #     'shape': data.shape,
        #     'pixel_size': self.pixel_size
        # }

        # Save processed TIFF
        tifffile.imwrite(self.output_path, data,
                         # metadata=serializable_metadata,
                         bigtiff=True)
        print(f"Processed TIFF saved to: {self.output_path}")

    def select_zslices(self, start=0, end=None):
        """
        Select Z-axis slice range
        :param start: Start slice index
        :param end: End slice index (None means the end)
        """
        if 'Z' not in self.axes:
            print("Warning: Current data does not contain a Z dimension")
            return

        max_z = self.shape[self.axes.index('Z')] - 1

        # Parameter validation
        if start < 0 or start > max_z:
            print(f"Error: Start slice index {start} is out of bounds [0-{max_z}]")
            return

        if end is not None and (end < start or end > max_z):
            print(f"Error: End slice index {end} is invalid")
            return

        self.z_start = start
        self.z_end = end or max_z
        print(f"Z-axis slice range set to: {start}-{self.z_end}")

    def _process_data(self):
        """Data preprocessing"""
        data = tifffile.imread(self.filepath)

        # Process time dimension
        if 'T' in self.axes:
            t_index = self.axes.index('T')
            data = data[self.t_start:self.t_end + 1]

        # Process Z-axis dimension
        if 'Z' in self.axes:
            z_index = self.axes.index('Z')
            # If z_end is not set, use the last slice
            z_end = self.z_end if self.z_end is not None else self.shape[z_index] - 1
            # If Z-axis is the first dimension, special handling is required
            if z_index == 0:
                data = data[self.z_start:z_end + 1]
            else:
                # Use take method to process non-first Z-axis
                data = data.take(indices=range(self.z_start, z_end + 1), axis=z_index)

        # Process channel dimension
        if 'C' in self.axes:
            c_index = self.axes.index('C')
            data = data.take(self.ch, axis=c_index)

        return data


if __name__ == "__main__":
    # Test cases
    test_dir = r'/subdata/r02c04'
    out_dir = r'/subdata/skeleton_result/tmp'
    all_paths = [f for f in os.listdir(test_dir) if f.endswith(('.tiff', '.tif'))]
    test_file = os.path.join(test_dir, all_paths)

    # Initialize processor
    file_info = Img_Information(test_file, output_dir=None)
    # Demonstrate functionality
    print("Original metadata:")
    print(f"Axis order: {file_info.axes}")

    # Modify settings and save
    try:
        # The actual area covered by the image is 200 μm × 200 μm, corresponding to 1080 × 1080 pixels.
        # The resolution in each direction is the actual size divided by the number of pixels: resolution = 200/1080 μm ≈ 0.1852 μm/pixel

        file_info.change_pixel_size('X', 0.185)
        file_info.change_pixel_size('Y', 0.185)
        file_info.create_save_tiff()

        print("\nGenerated paths:")
        for key, path in file_info.pipeline_paths.items():
            print(f"{key:>20}: {path}")

        # # Test memory mapping
        # preprocessed_memmap = im_info.get_memmap('im_preprocessed')
        # print("\nPreprocessed data shape:", preprocessed_memmap.shape)

    except Exception as e:
        print(f"\nOperation error: {str(e)}")


def save_processing_metadata(output_path, processing_params, file_dict, additional_info=None):
    """
    Save processing metadata to a txt file

    Parameters:
    - output_path: Output directory path
    - processing_params: Processing parameter configuration
    - file_dict: File dictionary {File ID: (File path, Mask path)}
    - additional_info: Other info to save
    """
    # Create metadata file path
    metadata_file = os.path.join(output_path, "processing_metadata.txt")

    # Prepare serializable parameter copies (handle non-serializable objects like numpy arrays)
    serializable_params = copy.deepcopy(processing_params)

    # Convert numpy arrays to lists
    for key, value in serializable_params.items():
        if isinstance(value, np.ndarray):
            serializable_params[key] = value.tolist()
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_params[key][sub_key] = sub_value.tolist()

    with open(metadata_file, 'w', encoding='utf-8') as f:
        # Write title and timestamp
        f.write("=" * 80 + "\n")
        f.write("Cytoskeleton Feature Extraction Processing Metadata\n")
        f.write("=" * 80 + "\n\n")

        # Basic Info
        f.write("1. Basic Processing Info\n")
        f.write("-" * 40 + "\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Total files: {len(file_dict)}\n")
        f.write(f"Parallel worker processes: {processing_params.get('image_level_max_workers', 'N/A')}\n\n")

        # File list information
        f.write("2. Processing File List\n")
        f.write("-" * 40 + "\n")
        for i, (file_id, paths) in enumerate(file_dict.items(), 1):
            f.write(f"{i:2d}. {file_id}:\n")

            # Check paths type and length
            if isinstance(paths, (tuple, list)):
                if len(paths) >= 2:
                    # At least 2 elements: image file and mask file
                    f.write(f"    Image file: {paths}\n")
                    f.write(f"    Mask file: {paths}\n")

                    # If more than 2 elements, output extra paths
                    if len(paths) > 2:
                        for j, extra_path in enumerate(paths[2:], 3):
                            # Convert extra_path directly to string
                            extra_path_str = str(extra_path)
                            if any(ext in extra_path_str for ext in ['.tif', '.tiff', '.png']):
                                f.write(f"    Extra path {j - 2}: {extra_path_str}\n")
                elif len(paths) == 1:
                    # Only 1 element: only image file
                    f.write(f"    Image file: {paths}\n")
                    f.write(f"    Mask file: None\n")
                else:
                    # Empty tuple or list
                    f.write(f"    Image file: None\n")
                    f.write(f"    Mask file: None\n")
            else:
                # paths is not tuple or list, output directly
                f.write(f"    Path: {paths}\n")

        f.write("\n")

        # Processing parameter configuration
        f.write("3. Processing Parameter Configuration\n")
        f.write("-" * 40 + "\n")

        # Main processing pipeline parameters
        f.write("Processing pipeline methods:\n")
        f.write(f"  Preprocessing method: {', '.join(serializable_params.get('preprocess_method', []))}\n")
        f.write(f"  Output method: {', '.join(serializable_params.get('output_method', []))}\n")
        f.write(f"  Selected levels: {', '.join(serializable_params.get('layer_select', []))}\n\n")

        # Smoothing parameters
        f.write("Smoothing parameters:\n")
        f.write(f"  Smoothing method: {serializable_params.get('smooth_method', 'N/A')}\n")
        f.write(f"  Gaussian sigma: {serializable_params.get('gaussian_sigma', 'N/A')}\n")
        f.write(f"  Median filter size: {serializable_params.get('median_size', 'N/A')}\n")
        f.write(f"  Top-hat transform radius: {serializable_params.get('tophat_radius', 'N/A')}\n")
        f.write(f"  Pre-threshold zeroing: {serializable_params.get('pre_threshold_zeroing', 'N/A')}\n")
        f.write(f"  Zeroing threshold: {serializable_params.get('zeroing_threshold', 'N/A')}\n\n")

        # Tubular structure detection parameters
        f.write("Tubular structure detection parameters:\n")
        f.write(f"  Minimum radius (pixels): {serializable_params.get('min_radius_px', 'N/A')}\n")
        f.write(f"  Maximum radius (pixels): {serializable_params.get('max_radius_px', 'N/A')}\n")
        f.write(f"  Ridge detection method: {serializable_params.get('ridge_method', 'N/A')}\n")
        f.write(f"  Max iterations: {serializable_params.get('max_iter', 'N/A')}\n\n")

        # Binarization parameters
        f.write("Binarization parameters:\n")
        f.write(f"  Binarization strategy: {serializable_params.get('binary_strategy', 'N/A')}\n")
        f.write(f"  Binarization method: {serializable_params.get('binary_method', 'N/A')}\n")
        f.write(f"  Threshold correction factor: {serializable_params.get('threshold_correction_factor', 'N/A')}\n")
        f.write(f"  Log transform: {serializable_params.get('log_transform', 'N/A')}\n\n")

        # Quality control parameters
        f.write("Quality control parameters:\n")
        f.write(f"  Minimum SNR: {serializable_params.get('min_snr', 'N/A')}\n")
        f.write(f"  Minimum object size: {serializable_params.get('min_object_size', 'N/A')}\n")
        f.write(f"  Minimum branch length: {serializable_params.get('min_branch_length', 'N/A')}\n")
        f.write(f"  Pruning iterations: {serializable_params.get('prune_iterations', 'N/A')}\n\n")

        # Network analysis parameters
        f.write("Network analysis parameters:\n")
        f.write(f"  Network angle threshold: {serializable_params.get('network_angle_thresh', 'N/A')}\n")
        f.write(f"  Network width threshold: {serializable_params.get('network_width_thresh', 'N/A')}\n")
        f.write(
            f"  Network distance threshold ratio: {serializable_params.get('network_dist_thresh_ratio', 'N/A')}\n\n")

        # Texture feature parameters
        f.write("Texture feature parameters:\n")
        f.write(f"  Haralick distance: {serializable_params.get('haralick_distance', 'N/A')}\n")
        f.write(f"  Haralick gray levels: {serializable_params.get('haralick_gray_levels', 'N/A')}\n\n")

        # Detailed ridge detection parameters
        f.write("Detailed ridge detection parameters:\n")
        ridge_params = serializable_params.get('ridge_params', {})
        for method, params in ridge_params.items():
            f.write(f"  {method}:\n")
            for param, value in params.items():
                if param == 'sigmas' and isinstance(value, list):
                    f.write(f"    {param}: [{value:.1f} ~ {value[-1]:.1f}] (total {len(value)} values)\n")
                else:
                    f.write(f"    {param}: {value}\n")
            f.write("\n")

        # Full JSON format parameters (for debugging)
        f.write("4. Full Parameter Configuration (JSON format)\n")
        f.write("-" * 40 + "\n")
        try:
            json.dump(serializable_params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            f.write(f"Parameter serialization error: {str(e)}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Metadata file generation complete\n")
        f.write("=" * 80 + "\n")

    print(f"Processing metadata saved to: {metadata_file}")
    return metadata_file