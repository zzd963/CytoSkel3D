import os
import pandas as pd
import shutil
from pathlib import Path


# Safer version, adding more error handling
def rename_and_copy_cell_files_safe():
    try:
        # Define paths
        source_base_path = r"\hipsc_single_edge_cell"
        target_base_path = r"\hipsc_single_edge_cell"

        # Check if the source directory exists
        if not os.path.exists(source_base_path):
            print(f"Error: Source directory does not exist - {source_base_path}")
            return

        # Check if metadata.csv exists
        metadata_path = os.path.join(source_base_path, "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Error: metadata.csv does not exist - {metadata_path}")
            return

        # Read metadata.csv
        print("Reading metadata.csv...")
        df = pd.read_csv(metadata_path)
        print(f"Found {len(df)} cell records")

        # Check if required columns exist
        required_columns = ['CellId', 'crop_raw', 'crop_seg']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: metadata.csv is missing required columns: {missing_columns}")
            return

        # Create target directories
        os.makedirs(target_base_path, exist_ok=True)
        os.makedirs(os.path.join(target_base_path, 'crop_raw'), exist_ok=True)
        os.makedirs(os.path.join(target_base_path, 'crop_seg'), exist_ok=True)
        print(f"Target directory created: {target_base_path}")

        # Process files
        success_count = 0
        error_count = 0

        for index, row in df.iterrows():
            try:
                cell_id = str(row['CellId']).strip()

                # Build full source file paths
                raw_source_path = os.path.join(source_base_path, str(row['crop_raw']).strip())
                seg_source_path = os.path.join(source_base_path, str(row['crop_seg']).strip())

                # Build target file paths
                raw_target_path = os.path.join(target_base_path, 'crop_raw', f"{cell_id}_raw.tif")
                seg_target_path = os.path.join(target_base_path, 'crop_seg', f"{cell_id}_seg.tif")

                # Check and copy raw image files
                if os.path.exists(raw_source_path):
                    shutil.copy2(raw_source_path, raw_target_path)
                else:
                    print(f"Warning: Raw file does not exist - {raw_source_path}")
                    error_count += 1
                    continue

                # Check and copy segmentation files
                if os.path.exists(seg_source_path):
                    shutil.copy2(seg_source_path, seg_target_path)
                else:
                    print(f"Warning: Segmentation file does not exist - {seg_source_path}")
                    # If the segmentation file does not exist, delete the copied raw file to maintain consistency
                    if os.path.exists(raw_target_path):
                        os.remove(raw_target_path)
                    error_count += 1
                    continue

                success_count += 1
                if success_count % 100 == 0:  # Output progress every 100 files
                    print(f"Processed {success_count} files...")

            except Exception as e:
                print(f"Error processing CellId {cell_id}: {str(e)}")
                error_count += 1

        # Output summary
        print(f"\n=== Processing Complete ===")
        print(f"Total records: {len(df)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {error_count}")
        print(f"Target directory: {target_base_path}")

    except Exception as e:
        print(f"Program execution error: {str(e)}")


if __name__ == "__main__":
    # Run the safe version of the function
    rename_and_copy_cell_files_safe()