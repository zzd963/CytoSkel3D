import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import copy
import math
import re
from glob import glob
from collections import defaultdict
from CytoSkel3D.preprocess.filter import HybridCytoskeletonProcessor
from CytoSkel3D.information.Img_Information import Img_Information, save_processing_metadata
from CytoSkel3D.analysis.skeleton import CytoskeletonAnalyzer3D
import tifffile


def ceil_to_half(value):
    """Round up to the nearest multiple of 0.5"""
    return math.ceil(value * 2) / 2


def run(img_info, params, outdir, file_id, binary_image=None):
    # If binary_image is provided, pass it to the processor
    if binary_image is not None:
        processor = HybridCytoskeletonProcessor(img_info, params, binary_image=binary_image)
    else:
        processor = HybridCytoskeletonProcessor(img_info, params)

    intermediates = processor.process_pipeline(visualize=True, save_intermediates=False, visualize_method=None)
    print("Processing complete, results saved in:", outdir)

    analyzer = CytoskeletonAnalyzer3D(img_info, intermediate=intermediates, params=params)
    analyzer.analyze_objects(debug_visualization=False, save_restruct=False)

    # Get multi-level feature reports
    report_data = analyzer.generate_report(visualize=True)

    # Add file ID to feature tables of all levels
    for level, df in report_data.items():
        if not df.empty:

            df.insert(0, 'file_id', file_id)
            # df['file_id'] = file_id
        else:
            print(f"Warning: Feature table for level {level} is empty, skipping adding file ID")

    return report_data


def scan_aics_cytoskel_files(data_root):
    """
    Scan all TIFF files in the AICS_hipsc_cytoskel dataset directory.
    Returns a dictionary formatted as: {file_id: (c2_raw_path, c2_seg_path, c4_seg_path, location, structure, cell_id)}
    """
    file_dict = {}
    locations = ["edge", "nonedge"]

    for location in locations:
        location_path = os.path.join(data_root, f"hipsc_single_{location}_cell")
        metadata_path = os.path.join(location_path, "metadata.csv")

        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file does not exist: {metadata_path}")
            continue

        # Read metadata file
        metadata = pd.read_csv(metadata_path)
        print(f"Found {len(metadata)} cells at {location} location")

        # Define cytoskeleton structures
        # cytoskeleton_structures = ["TUBA1B"]  # "TUBA1B", "ACTB", "ACTN1"
        cytoskeleton_structures = ["ACTB", "TUBA1B"]  # "TUBA1B", "ACTB", "ACTN1"

        # Filter cells with the target cytoskeleton structures
        cytoskeleton_cells = metadata[metadata['structure_name'].isin(cytoskeleton_structures)]
        print(f"Found {len(cytoskeleton_cells)} cytoskeleton structure cells at {location} location")

        for index, row in cytoskeleton_cells.iterrows():
            cell_id = row['CellId']
            structure = row['structure_name']

            # Process raw files for the c2 channel
            c2_raw_source = os.path.join(location_path, "crop_raw", structure, f"{cell_id}_c2_raw.tif")

            # Process seg files for the c2 channel
            c2_seg_source = os.path.join(location_path, "crop_seg", structure, f"{cell_id}_c2_seg.tif")

            # New: Process seg files for the c4 channel
            c4_seg_source = os.path.join(location_path, "crop_seg", structure, f"{cell_id}_c4_seg.tif")

            # Check if files exist
            if not os.path.exists(c2_raw_source):
                print(f"Warning: c2_raw file does not exist - {c2_raw_source}")
                continue

            if not os.path.exists(c2_seg_source):
                print(f"Warning: c2_seg file does not exist - {c2_seg_source}")
                continue

            # c4_seg file is optional, set to None if it does not exist
            if not os.path.exists(c4_seg_source):
                print(f"Warning: c4_seg file does not exist - {c4_seg_source}, will not use pre-segmentation results")
                c4_seg_source = None

            # File ID format: {location}_{cell_id}_{structure}
            file_id = f"{location}_{cell_id}_{structure}"

            file_dict[file_id] = (c2_raw_source, c2_seg_source, c4_seg_source, location, structure, cell_id)
            print(f"Found file: {file_id} -> {c2_raw_source}")

    return file_dict


def process_single_file(file_info, output_root):
    """Independent function for processing a single image file, used for parallel execution"""
    file_id, paths_info = file_info
    file_path, mask_path, c4_seg_path, location, structure, cell_id = paths_info

    print(f"Processing file: {file_id}")
    print(f"Raw file: {file_path}")
    print(f"Mask file: {mask_path}")
    if c4_seg_path:
        print(f"Cytoskeleton mask file: {c4_seg_path}")
    print(f"Location: {location}, Structure: {structure}")

    try:
        # Build output directory path
        out_dir = os.path.join(output_root, location, structure, str(cell_id))

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'features'), exist_ok=True)

        # Read c4_seg image as binary_image (if it exists)
        binary_image = None
        if c4_seg_path and os.path.exists(c4_seg_path):
            try:
                binary_image = tifffile.imread(c4_seg_path)
                print(f"Successfully read c4_seg image, shape: {binary_image.shape}, dtype: {binary_image.dtype}")

                # Ensure the image is a binary image (0 and 1)
                if binary_image.dtype != bool:
                    # Binarize the image if it is not boolean
                    binary_image = binary_image > 0
                    print("Converted c4_seg image to binary image")

            except Exception as e:
                print(f"Error reading c4_seg image: {str(e)}")
                binary_image = None

        # Create image information object and pass the mask path
        img_info = Img_Information(file_path, output_dir=out_dir, maskpath=mask_path)
        img_info.create_save_tiff()

        # Set processing parameters
        processing_params = copy.deepcopy(PROCESSING_PARAMS)

        # Run processing pipeline, passing the binary_image parameter
        df_dict = run(
            img_info,
            params=processing_params,
            outdir=out_dir,
            file_id=file_id,
            binary_image=binary_image  # Pass c4_seg image as pre-segmentation result
        )

        return df_dict

    except Exception as e:
        print(f"Error occurred while processing {file_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Parameter configuration - Adjusted for the AICS dataset
PROCESSING_PARAMS = {
    'preprocess_method': [
        'smooth',
        'ridge',
        'binary',
        'skeleton',
        'post_processing'
    ],

    # Smoothing method parameters
    'smooth_method': 'gaussian',
    'gaussian_sigma': 1.0,
    'median_size': 3,
    'tophat_radius': 15,

    # Background zeroing parameters
    'pre_threshold_zeroing': True,
    'zeroing_threshold': 0.4,

    # Tubular structure size parameters
    'min_radius_px': 1,
    'max_radius_px': 5,

    # Ridge enhancement parameters
    'use_radius_for_sigma': True,
    'ridge_method': 'Frangi',
    'max_iter': 3,
    'ridge_params': {
        'Sato': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Frangi': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Frangi_iteration': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'Meijering': {
            'sigmas': np.linspace(1, 4, 7),
            'black_ridges': False
        },
        'multi_orient_skel': {
            'angles': [0, 30, 60, 90, 120, 150],
            'sigmas': np.linspace(1, 4, 7)
        }
    },

    # Binarization parameters
    'binary_strategy': 'Global',
    'binary_method': 'multiotsu',
    'threshold_correction_factor': 1.0,
    'log_transform': False,
    'assign_middle_to_foreground': "background",
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
        'smooth',
        'ridge',
        'binary',
        'skeleton',
        'post_processing'
    ],

    # Anisotropic processing parameters
    'apply_anisotropic_scaling': False,
    'z_scale_mode': 'fixed',
    'fixed_z_scale': 1.0,
    'voxel_size': (1.0, 1.0, 1.0),

    'full_image_mode': False,
    'layer_select': ['nodes', 'segments', 'branches', 'network', 'cell'],
    'if_restruct': False,

    'network_angle_thresh': 30,
    'network_width_thresh': 0.3,
    'network_dist_thresh_ratio': 3,

    # Texture feature parameters
    'skeleton_distance': [1, 3, 5],
    'raw_distance': [5, 10, 20],

    'image_level_max_workers': 14,
    'object_level_max_workers': 1,
}

if __name__ == "__main__":
    # Modify to the new AICS dataset path (path after channel separation)
    data_root = r"AICS_hipsc_cytoskel"
    output_path = r"\AICS_hipsc_cytoskel\edge_nonedge"

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create output directories for each location and structure
    locations = ["edge", "nonedge"]
    structures = ["ACTB", "TUBA1B"]  # "TUBA1B", "ACTB", "ACTN1"

    for location in locations:
        for structure in structures:
            os.makedirs(os.path.join(output_path, location, structure), exist_ok=True)

    # Scan AICS cytoskeleton dataset files
    file_dict = scan_aics_cytoskel_files(data_root)
    num_files = len(file_dict)
    print(f"Total of {num_files} AICS cytoskeleton files found:")
    for i, (file_id, info) in enumerate(file_dict.items()):
        print(f"[{i + 1}/{num_files}] {file_id}")

    # Save metadata before processing begins
    metadata_file = save_processing_metadata(output_path, PROCESSING_PARAMS, file_dict)

    # Create task list
    tasks = [
        (file_info, output_path)
        for file_info in file_dict.items()
    ]

    # Process in parallel using ProcessPoolExecutor
    all_reports = []

    top_level_max_workers = PROCESSING_PARAMS['image_level_max_workers']
    print(f"Processing image files in parallel using {top_level_max_workers} processes...")

    with ProcessPoolExecutor(max_workers=top_level_max_workers) as executor:
        futures = {
            executor.submit(process_single_file, *task): task[0][0]
            for task in tasks
        }

        for future in as_completed(futures):
            file_id = futures[future]
            try:
                report_dict = future.result()
                if report_dict:
                    all_reports.append((file_id, report_dict))
                    print(f"Successfully processed: {file_id}")
                else:
                    print(f"Warning: No valid feature data for {file_id}")
            except Exception as e:
                print(f"Critical error occurred while processing {file_id}: {str(e)}")

    # ===================== Save merged feature tables for all location conditions grouped by structure =====================
    # Group features by structure
    structure_reports = defaultdict(lambda: defaultdict(list))

    for file_id, report in all_reports:
        # Parse file ID: {location}_{cell_id}_{structure}
        parts = file_id.split('_')
        location = parts[0]  # edge or nonedge
        cell_id = parts[1]  # cell ID
        structure = '_'.join(parts[2:])  # structure name

        # Iterate through each level in the report
        for level, df in report.items():
            if not df.empty:
                # Add location and structure information
                df['location'] = location
                df['structure'] = structure
                df['cell_id'] = cell_id
                structure_reports[structure][level].append(df)

    # Save feature tables for each structure (merging all locations)
    for structure, level_dict in structure_reports.items():
        for level, df_list in level_dict.items():
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)

                # Generate new column names (add prefix to non-protected columns)
                protected_columns = ['file_id', 'object_id', 'location', 'structure', 'cell_id']
                new_columns = [
                    f"Skeleton_{col}" if col not in protected_columns else col
                    for col in combined_df.columns
                ]
                combined_df.columns = new_columns

                # Save the final results
                csv_output_path = os.path.join(output_path, f'all_{level}_features_{structure}.csv')
                combined_df.to_csv(csv_output_path, index=False)
                print(f"Saved {level} features for {structure} structure to: {csv_output_path}")

                # Print statistics
                print(f"{structure} - {level} Feature Statistics:")
                print(f"  Total samples: {len(combined_df)}")
                print(f"  Location distribution: {combined_df['location'].value_counts().to_dict()}")
            else:
                print(f"Warning: No valid data for level {level} of {structure} structure, skipping save")

    print(f"\nAICS cytoskeleton dataset feature extraction complete! Processed {len(all_reports)} files in total")