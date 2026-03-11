# Goal: Master scheduler for 3D cytoskeleton analysis


import os
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import matplotlib as mpl
from skimage import morphology, measure

from CytoSkel3D.information.Img_Information import Img_Information
from .CytoskeletonNetwork_Analyzer import CytoskeletonNetworkAnalyzer
from .Load_Segmentation import Load_Segmentation
from .FeatureExtractor import FeatureExtractor
from .VisualizationReporter import VisualizationReporter, VisualizationManager


class CytoskeletonAnalyzer3D:
    """Initialize 3D cytoskeleton analyzer"""

    def __init__(self, img_info, intermediate=None, params=None):
        self.img_info = img_info
        self.params = params.copy() if params else {}
        self.layer_select = self.params.get('layer_select', None)

        # Initialize core data
        if not intermediate:
            self.raw_image = self._load_verified_data('processed', dtype=np.float32)
            try:
                self.ridge_image = self._load_verified_data('im_pre_ridge', dtype=bool)
            except:
                self.ridge_image = None
            try:
                self.binary_image = self._load_verified_data('im_pre_binary', dtype=bool)
            except:
                self.binary_image = None
            self.skeleton = self._load_verified_data('im_pre_cleaned_skeleton', dtype=bool)
            self.pixel_class = self._load_verified_data('im_pre_cleaned_skeleton_pixel_class', dtype=np.uint8)
            self.labeled_skeleton = self._load_verified_data('im_pre_cleaned_skeleton_relabeled', dtype=np.uint16)
        else:
            self.raw_image = intermediate.get('processed', self._load_verified_data('processed', dtype=np.float32))
            self.ridge_image = intermediate.get('ridge', None)
            self.binary_image = intermediate.get('binary', None)
            self.skeleton = intermediate['post_processing']
            self.pixel_class = intermediate['post_processing']
            self.labeled_skeleton = intermediate['post_processing']

        self.is_3d = self.skeleton.ndim == 3
        self._load_mask()

        # === Anisotropy and voxel size processing (Fixed version) ===
        self.apply_anisotropic = self.params.get('apply_anisotropic_scaling', False)
        default_vs = (1.0, 1.0, 1.0) if self.is_3d else (1.0, 1.0)

        # 1. Control logic: determine data source based on flag
        if self.apply_anisotropic:
            vs = self.params.get('voxel_size', default_vs)
            # Optional: Add strict dimension check warning here
            expected_dims = 3 if self.is_3d else 2
            if len(vs) != expected_dims:
                print(f"Warning: Image is {expected_dims}D, but provided voxel_size is {len(vs)}D. Will correct automatically.")
        else:
            vs = default_vs

        # 2. Data structure logic: rigorous dimension completion (uniformly converted to z, y, x)
        vs_arr = np.array(vs, dtype=float)
        if len(vs_arr) == 2:
            # 2D input (y, x) -> convert to (1.0, y, x) for uniform calculation
            self.voxel_size = np.insert(vs_arr, 0, 1.0)
        elif len(vs_arr) == 3:
            # 3D input (z, y, x) -> keep unchanged
            self.voxel_size = vs_arr
        else:
            # Fallback for abnormal cases
            self.voxel_size = np.array([1.0, 1.0, 1.0])

        # Cache attributes
        self.network_cache = {}
        self.feature_table = []

        self._configure_matplotlib()
        self.viz_lock = Lock()
        self.file_lock = Lock()
        self.reporter = None  # Initialize reporter

    def _configure_matplotlib(self):
        mpl.rcParams['axes.formatter.use_mathtext'] = False
        mpl.rcParams['axes.formatter.limits'] = (-5, 5)

    def _load_verified_data(self, data_key, dtype):
        data = self.img_info.get_memmap(data_key)[:].astype(dtype)
        if data.ndim not in (2, 3):
            raise ValueError(f"Data dimension error: {data_key} should be a 2D/3D array")
        return data

    def _load_mask(self):
        loader = Load_Segmentation(img_info=self.img_info, skeleton=self.skeleton, params=self.params)
        self.object_mask, self.labeled_objects = loader._init_mask_system()
        self.full_image_mode = loader.full_image_mode

    def analyze_objects(self, debug_visualization=False, save_restruct=False):
        """Process multiple objects in parallel"""
        object_ids = np.unique(self.labeled_objects)
        object_ids = object_ids[object_ids != 0]

        if self.full_image_mode and len(object_ids) == 0:
            object_ids = np.array()

        max_workers = self.params.get('object_level_max_workers', 1)
        debug_visualization = debug_visualization and (max_workers == 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for obj_id in object_ids:
                if self.full_image_mode:
                    obj_mask = self.labeled_objects == obj_id if obj_id in self.labeled_objects else np.ones_like(
                        self.labeled_objects, dtype=bool)
                else:
                    obj_mask = self.labeled_objects == obj_id

                if not self._validate_object(obj_mask):
                    continue
                futures.append(
                    executor.submit(self._analyze_single_object, obj_mask, obj_id, debug_visualization, save_restruct))

            self.feature_table = [f.result() for f in futures if f.result() is not None]

    def _validate_object(self, obj_mask):
        min_voxels = self.params['min_object_size']
        return np.sum(obj_mask) >= min_voxels

    def _analyze_single_object(self, obj_mask, obj_id, debug_visualization=False, save_restruct=False):
        """Single object analysis pipeline"""
        skel_obj = self.skeleton & obj_mask
        raw_obj = self.raw_image * obj_mask

        try:
            # Execute network analysis (get intermediate results)
            build_results = self._analyze_network(skel_obj, raw_obj, obj_mask, obj_id, save_restruct,
                                                  debug_visualization)

            if not hasattr(self, 'analyzer') or not self.analyzer or not hasattr(self.analyzer, 'graph'):
                raise RuntimeError("Network analyzer not properly initialized")

            # Feature extraction (suggest decoupling in the future: pass analyzer.graph directly)
            self.extractor = FeatureExtractor(
                graph=self.analyzer.graph,
                edge_properties=self.analyzer.edge_properties,
                centroids=self.analyzer.centroids,
                skeleton=self.analyzer.skeleton,
                raw_image=raw_obj,
                ridge_image=self.ridge_image,
                obj_mask=obj_mask,
                voxel_size=self.voxel_size,
                params=self.params,
                layer_select=self.layer_select,
                full_image_mode=self.full_image_mode,
                is_3d = self.is_3d  # Explicitly pass dimension information
            )

            # FeatureExtractor(self, layer_select=self.layer_select)
            feature_results = self.extractor._calc_network_features()

            # Save to cache
            self.network_cache[obj_id] = self.analyzer

            return {
                'object_id': obj_id,
                'nodes': feature_results['nodes'],
                'segments': feature_results['segments'],
                'branches': feature_results['branches'],
                'network': feature_results['network'],
                'cell': feature_results['cell']
            }
        except Exception as e:
            print(f"Reason for failure of object {obj_id} analysis: {str(e)}")
            traceback.print_exc()
            return None

    def _analyze_network(self, skeleton, raw, object_mask, obj_id, save_restruct, debug_visualization):
        """Execute object-level network analysis"""
        try:
            if np.sum(skeleton) < 10: return None

            # 1. Pure calculation: initialize and run analyzer
            self.analyzer = CytoskeletonNetworkAnalyzer(
                params=self.params,
                img_info=self.img_info,
                object_id=obj_id,
                skeleton=skeleton,
                raw_image=raw,
                object_mask=object_mask,
                pixel_class=self.pixel_class,
                labeled_skeleton=self.labeled_skeleton,
                ridge_image=self.ridge_image,
                voxel_size=self.voxel_size,
                full_image_mode=self.full_image_mode
            )

            # Get intermediate results of network construction (segments_bp, segments_connected)
            build_results = self.analyzer.analyze_network(save_restruct=save_restruct)

            # 2. Pure display: if debug visualization is enabled, call Reporter
            if debug_visualization:
                # Ensure reporter exists and is associated with current analyzer
                self.reporter = VisualizationReporter(self)

                # Prepare debug directory
                debug_vis_path = os.path.join(self.img_info.graph_dir, "debug_vis")
                os.makedirs(debug_vis_path, exist_ok=True)

                # Call newly migrated method: visualize network construction process
                self.reporter.visualize_network_construction(
                    build_results['segments_bp'],
                    build_results['segments_connected']
                )

                # Call original visualization methods
                if self.is_3d:
                    bg_img = np.max(raw, axis=0)
                    self.reporter.visualize_3d_paths(debug_vis_path, obj_id)
                else:
                    bg_img = raw

                self.reporter.visualize_network_projection(debug_vis_path, obj_id, bg_img=bg_img)
                self.reporter.export_paths_as_tiff(debug_vis_path, obj_id)

            return build_results

        except Exception as e:
            print(f"Network analysis failed: {str(e)}")
            traceback.print_exc()
            return None

    def generate_report(self, visualize=True):
        """Generate analysis report"""
        report_data = self._compile_full_report()
        self._save_excel_report(report_data)
        self._save_csv_reports(report_data)

        if visualize:
            viz_manager = VisualizationManager(self)

            # 1. Overall network projection visualization
            viz_manager.visualize_global_network()

            # # 2. Overall 3D path visualization (only 3D images)
            # if self.is_3d:
            #     viz_manager.visualize_global_3d_paths()

        return report_data


    def _compile_full_report(self):
        report_data = {
            'nodes': [],
            'segments': [],
            'branches': [],
            'network': [],
            'cell': []
        }

        for obj_data in self.feature_table:
            for level in report_data.keys():
                df = obj_data[level].copy()
                df.insert(0, 'object_id', obj_data['object_id'])

                report_data[level].append(df)

        # Merge into complete DataFrame
        for level in report_data:
            report_data[level] = pd.concat(report_data[level])

        return report_data

    def _save_excel_report(self, report_data):
        """Save Excel report (multi-sheet)"""
        excel_path = os.path.join(self.img_info.feature_dir, 'full_analysis.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            for level, df in report_data.items():
                sheet_name = f"{level.capitalize()} Features"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _save_csv_reports(self, report_data):
        """Save CSV detail reports"""
        for level, df in report_data.items():
            csv_path = os.path.join(self.img_info.feature_dir, f"{level}_features.csv")
            df.to_csv(csv_path, index=False)

    def _save_analysis_results(self, feature_table):
        """Enhanced result cleaning and saving"""
        # Clean temporary fields
        # Clean temporary fields and segment information
        df = pd.DataFrame(feature_table)
        columns_to_drop = [
            col for col in df.columns
            if col.startswith('__') or col == 'lines_3d'
        ]

        clean_df = df.drop(columns=columns_to_drop, errors='ignore')

        # Add metadata
        metadata = {
            'voxel_size': [tuple(self.voxel_size)],  # Wrap as list
            'analysis_date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
            'object_count': [len(clean_df)]  # Wrap as list
        }

        # Save cleaned data
        stats = clean_df.describe().T
        with pd.ExcelWriter(os.path.join(self.img_info.feature_dir, 'skeleton_features.xlsx'),
                            engine='openpyxl') as writer:
            # Main data table
            clean_df.to_excel(writer, sheet_name='Object Features', index=False)

            # Statistics table
            stats.to_excel(writer, sheet_name='Summary Statistics')

            # Metadata table
            metadata_series = pd.Series(metadata)
            metadata_series.to_excel(
                writer,
                sheet_name='Metadata',
                header=['Value']
            )
        return clean_df


# Parameter configuration
PROCESSING_PARAMS = {
    'full_image_mode': False,  # Enable full image analysis mode
    'apply_anisotropic_scaling': False,  # Enable anisotropic processing
    'voxel_size': (0.185, 0.185),
    # 'voxel_size': (0.2, 0.12, 0.12,)  # z,y,x
    'min_object_size': 10,  # Minimum number of voxels

    'network_angle_thresh': 30,  # Segment merge angle threshold (degrees)
    'network_width_thresh': 0.3,  # Segment merge width threshold (ratio)
    'network_dist_thresh_ratio': 3,  # Distance threshold ratio factor

    'object_level_max_workers':4,
    'layer_select': ['nodes', 'segments', 'branches', 'network', 'cell']  # Calculate all by default
}



# Usage example
if __name__ == "__main__":
    test_dir =
    out_dir =
    # out_dir = r'E:\ZZD\Code\Cytoskeleton\Cytoskeleton\Cytoskeleton_3D\output\idr_50_10\ridge_method\sato'

    all_paths = [f for f in os.listdir(test_dir) if f.endswith(('.tiff', '.tif'))]
    test_file = os.path.join(test_dir, all_paths)

    mask_path = r'E:\ZZD\Code\Cytoskeleton\experiment\A250523_M230925\subdata\cellprofiler_result\mask\B04_01_01_001--cell.png'
    img_info = Img_Information(test_file, output_dir=out_dir, maskpath=mask_path)
    img_info.change_dim_res('X', 0.185)
    img_info.change_dim_res('Y', 0.185)

    analyzer = CytoskeletonAnalyzer3D(img_info, intermediate=None, params=PROCESSING_PARAMS)
    analyzer.analyze_objects(debug_visualization=False)

    analyzer.generate_report(out_dir)