import os
import pandas as pd
import shutil
from tifffile import imread, imwrite
import numpy as np
from tqdm import tqdm


def find_channel_dimension(shape, expected_channels):
    """
    Find the channel dimension in the image shape.
    Returns the index of the channel dimension, or None if not found.
    """
    for i, dim in enumerate(shape):
        if dim == expected_channels:
            return i
    return None


def separate_channels_and_copy(base_path, output_base):
    metadata_path = os.path.join(base_path, "metadata.csv")

    # Define cytoskeleton structures
    cytoskeleton_structures = ["TUBA1B", "ACTB", "ACTN1"]

    # Read metadata.csv
    print("Reading metadata.csv...")
    metadata = pd.read_csv(metadata_path)
    print(f"Total of {len(metadata)} cells in raw data")

    # Filter cells with cytoskeleton structures
    cytoskeleton_cells = metadata[metadata['structure_name'].isin(cytoskeleton_structures)]
    print(f"Found {len(cytoskeleton_cells)} cells with cytoskeleton structures")

    if len(cytoskeleton_cells) == 0:
        print("No matching cells found, program terminating")
        return

    # Create output directory structure
    raw_base_dir = os.path.join(base_path, "crop_raw")
    seg_base_dir = os.path.join(base_path, "crop_seg")

    raw_output_dir = os.path.join(output_base, "crop_raw")
    seg_output_dir = os.path.join(output_base, "crop_seg")

    # Create subdirectories for each structure
    for structure in cytoskeleton_structures:
        os.makedirs(os.path.join(raw_output_dir, structure), exist_ok=True)
        os.makedirs(os.path.join(seg_output_dir, structure), exist_ok=True)

    # Process files (with progress bar)
    print("Starting to process files...")
    processed_count = 0
    error_cells = []

    for index, row in tqdm(cytoskeleton_cells.iterrows(), total=len(cytoskeleton_cells), desc="Processing progress"):
        cell_id = row['CellId']
        structure = row['structure_name']

        # Source file paths
        raw_source = os.path.join(raw_base_dir, f"{cell_id}_raw.tif")
        seg_source = os.path.join(seg_base_dir, f"{cell_id}_seg.tif")

        try:
            # Process raw files: separate three channels
            if os.path.exists(raw_source):
                # Read multi-channel TIFF file
                raw_image = imread(raw_source)
                print(f"raw image shape: {raw_image.shape}")

                # Find channel dimension (dimension of size 3)
                channel_dim = find_channel_dimension(raw_image.shape, 3)

                if channel_dim is not None:
                    # Separate three channels
                    for channel_idx, channel_name in enumerate(['c0', 'c1', 'c2']):
                        # Use numpy.take to extract specific channels along the channel dimension
                        channel_data = np.take(raw_image, indices=channel_idx, axis=channel_dim)

                        channel_target = os.path.join(raw_output_dir, structure, f"{cell_id}_{channel_name}_raw.tif")
                        imwrite(channel_target, channel_data)

                    print(f"Successfully separated three channels for {cell_id} (channel dimension: {channel_dim})")
                else:
                    print(f"Warning: 3 channels not found in raw file for {cell_id}, actual shape {raw_image.shape}")
                    error_cells.append(cell_id)
                    continue
            else:
                print(f"Warning: Source file does not exist - {raw_source}")
                error_cells.append(cell_id)
                continue

            # Process seg files: take only the specified channel (entire cell mask)
            if os.path.exists(seg_source):
                # Read multi-channel TIFF file
                seg_image = imread(seg_source)
                print(f"seg image shape: {seg_image.shape}")

                # Find channel dimension (dimension of size 5)
                channel_dim = find_channel_dimension(seg_image.shape, 5)

                if channel_dim is not None:
                    # Take only the target channel (index 4)
                    cell_mask_channel = np.take(seg_image, indices=4, axis=channel_dim)

                    seg_target = os.path.join(seg_output_dir, structure, f"{cell_id}_c4_seg.tif")
                    imwrite(seg_target, cell_mask_channel)

                    print(f"Successfully extracted cell mask channel for {cell_id} (channel dimension: {channel_dim})")
                else:
                    print(f"Warning: 5 channels not found in seg file for {cell_id}, actual shape {seg_image.shape}")
                    error_cells.append(cell_id)
                    continue
            else:
                print(f"Warning: Source file does not exist - {seg_source}")
                error_cells.append(cell_id)
                continue

            processed_count += 1

        except Exception as e:
            print(f"Error processing cell {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            error_cells.append(cell_id)

    print(f"\nSuccessfully processed files for {processed_count} cells")
    if error_cells:
        print(f"Failed to process {len(error_cells)} cells: {error_cells}")

    # Save new metadata.csv
    new_metadata_path = os.path.join(output_base, "metadata.csv")
    cytoskeleton_cells.to_csv(new_metadata_path, index=False)
    print(f"New metadata.csv saved to: {new_metadata_path}")

    # Display statistical information
    print("\nStatistics by structure:")
    structure_counts = cytoskeleton_cells['structure_name'].value_counts()
    for structure, count in structure_counts.items():
        print(f"{structure}: {count} cells")

    # Display file information
    print("\nGenerated file structure:")
    print("crop_raw/structure_name/:")
    print("  - {cell_id}_c0_raw.tif (dna channel)")
    print("  - {cell_id}_c1_raw.tif (membrane channel)")
    print("  - {cell_id}_c2_raw.tif (structure channel)")
    print("crop_seg/structure_name/:")
    print("  - {cell_id}_c2_seg.tif (entire cell mask)")


def process_both_locations():
    """Process both edge and nonedge locations simultaneously"""
    locations = [
        ("edge", r"\hipsc_single_edge_cell",
         r"\hipsc_single_edge_cell"),
        ("nonedge", r"\hipsc_single_nonedge_cell",
         r"\hipsc_single_nonedge_cell")
    ]

    for location_name, base_path, output_base in locations:
        print(f"\n{'=' * 50}")
        print(f"Starting to process {location_name} location")
        print(f"{'=' * 50}")

        # Call processing function
        separate_channels_and_copy(base_path, output_base)


if __name__ == "__main__":
    # Can choose to process only one location or both locations
    process_both_locations()  # Process both locations