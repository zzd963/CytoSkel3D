# 目标：3D细胞骨架分析总调度器


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
    """3D细胞骨架分析器初始化"""

    def __init__(self, img_info, intermediate=None, params=None):
        self.img_info = img_info
        self.params = params.copy() if params else {}
        self.layer_select = self.params.get('layer_select', None)

        # 初始化核心数据
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
            self.skeleton = intermediate['post_processing'][0]
            self.pixel_class = intermediate['post_processing'][2]
            self.labeled_skeleton = intermediate['post_processing'][1]

        self.is_3d = self.skeleton.ndim == 3
        self._load_mask()

        # === 各向异性与体素尺寸处理 (修复版) ===
        self.apply_anisotropic = self.params.get('apply_anisotropic_scaling', False)
        default_vs = (1.0, 1.0, 1.0) if self.is_3d else (1.0, 1.0)

        # 1. 控制逻辑：根据 flag 决定数据源
        if self.apply_anisotropic:
            vs = self.params.get('voxel_size', default_vs)
            # 可选：在这里添加严格的维度检查警告
            expected_dims = 3 if self.is_3d else 2
            if len(vs) != expected_dims:
                print(f"警告: 图像是 {expected_dims}D，但提供的 voxel_size 是 {len(vs)}D。将自动修正。")
        else:
            vs = default_vs

        # 2. 数据结构逻辑：严谨的维度补全 (统一转换为 z, y, x)
        vs_arr = np.array(vs, dtype=float)
        if len(vs_arr) == 2:
            # 2D 输入 (y, x) -> 转换为 (1.0, y, x) 以便统一计算
            self.voxel_size = np.insert(vs_arr, 0, 1.0)
        elif len(vs_arr) == 3:
            # 3D 输入 (z, y, x) -> 保持不变
            self.voxel_size = vs_arr
        else:
            # 异常情况回退
            self.voxel_size = np.array([1.0, 1.0, 1.0])

        # 缓存属性
        self.network_cache = {}
        self.feature_table = []

        self._configure_matplotlib()
        self.viz_lock = Lock()
        self.file_lock = Lock()
        self.reporter = None  # 初始化 reporter

    def _configure_matplotlib(self):
        mpl.rcParams['axes.formatter.use_mathtext'] = False
        mpl.rcParams['axes.formatter.limits'] = (-5, 5)

    def _load_verified_data(self, data_key, dtype):
        data = self.img_info.get_memmap(data_key)[:].astype(dtype)
        if data.ndim not in (2, 3):
            raise ValueError(f"数据维度错误: {data_key} 应为2D/3D数组")
        return data

    def _load_mask(self):
        loader = Load_Segmentation(img_info=self.img_info, skeleton=self.skeleton, params=self.params)
        self.object_mask, self.labeled_objects = loader._init_mask_system()
        self.full_image_mode = loader.full_image_mode

    def analyze_objects(self, debug_visualization=False, save_restruct=False):
        """并行处理多个对象"""
        object_ids = np.unique(self.labeled_objects)
        object_ids = object_ids[object_ids != 0]

        if self.full_image_mode and len(object_ids) == 0:
            object_ids = np.array([1])

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
        """单个对象分析流程"""
        skel_obj = self.skeleton & obj_mask
        raw_obj = self.raw_image * obj_mask

        try:
            # 执行网络分析（获取中间结果）
            build_results = self._analyze_network(skel_obj, raw_obj, obj_mask, obj_id, save_restruct,
                                                  debug_visualization)

            if not hasattr(self, 'analyzer') or not self.analyzer or not hasattr(self.analyzer, 'graph'):
                raise RuntimeError("网络分析器未正确初始化")

            # 特征提取 (建议未来解耦：直接传入 analyzer.graph)
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
                is_3d = self.is_3d  # 显式传递维度信息
            )

            # FeatureExtractor(self, layer_select=self.layer_select)
            feature_results = self.extractor._calc_network_features()

            # 保存到缓存
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
            print(f"对象 {obj_id} 分析失败原因: {str(e)}")
            traceback.print_exc()
            return None

    def _analyze_network(self, skeleton, raw, object_mask, obj_id, save_restruct, debug_visualization):
        """执行对象级网络分析"""
        try:
            if np.sum(skeleton) < 10: return None

            # 1. 纯计算：初始化并运行分析器
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

            # 获取网络构建的中间结果 (segments_bp, segments_connected)
            build_results = self.analyzer.analyze_network(save_restruct=save_restruct)

            # 2. 纯展示：如果启用了调试可视化，调用 Reporter
            if debug_visualization:
                # 确保 reporter 存在且关联当前 analyzer
                self.reporter = VisualizationReporter(self)

                # 准备调试目录
                debug_vis_path = os.path.join(self.img_info.graph_dir, "debug_vis")
                os.makedirs(debug_vis_path, exist_ok=True)

                # 调用新迁移的方法：可视化网络构建过程
                self.reporter.visualize_network_construction(
                    build_results['segments_bp'],
                    build_results['segments_connected']
                )

                # 调用原有的可视化方法
                if self.is_3d:
                    bg_img = np.max(raw, axis=0)
                    self.reporter.visualize_3d_paths(debug_vis_path, obj_id)
                else:
                    bg_img = raw

                self.reporter.visualize_network_projection(debug_vis_path, obj_id, bg_img=bg_img)
                self.reporter.export_paths_as_tiff(debug_vis_path, obj_id)

            return build_results

        except Exception as e:
            print(f"网络分析失败: {str(e)}")
            traceback.print_exc()
            return None

    def generate_report(self, visualize=True):
        """生成分析报告"""
        report_data = self._compile_full_report()
        self._save_excel_report(report_data)
        self._save_csv_reports(report_data)

        if visualize:
            viz_manager = VisualizationManager(self)

            # 1. 整体网络投影可视化
            viz_manager.visualize_global_network()

            # # 2. 整体3D路径可视化（仅3D图像）
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

        # 合并为完整DataFrame
        for level in report_data:
            report_data[level] = pd.concat(report_data[level])

        return report_data

    def _save_excel_report(self, report_data):
        """保存Excel报告（多sheet）"""
        excel_path = os.path.join(self.img_info.feature_dir, 'full_analysis.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            for level, df in report_data.items():
                sheet_name = f"{level.capitalize()} Features"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _save_csv_reports(self, report_data):
        """保存CSV明细报告"""
        for level, df in report_data.items():
            csv_path = os.path.join(self.img_info.feature_dir, f"{level}_features.csv")
            df.to_csv(csv_path, index=False)

    def _save_analysis_results(self, feature_table):
        """增强型结果清洗与保存"""
        # 清洗临时字段
        # 清洗临时字段和线段信息
        df = pd.DataFrame(feature_table)
        columns_to_drop = [
            col for col in df.columns
            if col.startswith('__') or col == 'lines_3d'
        ]

        clean_df = df.drop(columns=columns_to_drop, errors='ignore')

        # 添加元数据
        metadata = {
            'voxel_size': [tuple(self.voxel_size)],  # 包装成列表
            'analysis_date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
            'object_count': [len(clean_df)]  # 包装成列表
        }

        # 保存清洗后的数据
        stats = clean_df.describe().T
        with pd.ExcelWriter(os.path.join(self.img_info.feature_dir, 'skeleton_features.xlsx'),
                            engine='openpyxl') as writer:
            # 主数据表
            clean_df.to_excel(writer, sheet_name='Object Features', index=False)

            # 统计表
            stats.to_excel(writer, sheet_name='Summary Statistics')

            # 元数据表
            metadata_series = pd.Series(metadata)
            metadata_series.to_excel(
                writer,
                sheet_name='Metadata',
                header=['Value']
            )
        return clean_df


# 参数配置
PROCESSING_PARAMS = {
    'full_image_mode': False,  # 启用全图分析模式
    'apply_anisotropic_scaling': False,  # 启用各向异性处理
    'voxel_size': (0.185, 0.185),
    # 'voxel_size': (0.2, 0.12, 0.12,)  # z,y,x
    'min_object_size': 10,  # 最小体素数

    'network_angle_thresh': 30,  # 片段合并角度阈值(度)
    'network_width_thresh': 0.3,  # 片段合并宽度阈值(比例)
    'network_dist_thresh_ratio': 3,  # 距离阈值比例因子

    'object_level_max_workers':4,
    'layer_select': ['nodes', 'segments', 'branches', 'network', 'cell']  # 默认全部计算
}
'''
1. numpeaks: 400
功能：
    控制每层（Z轴切片）允许检测的最大线段数量
工作机制  ：
    霍夫变换后，在参数空间（θ,ρ）中按得票数降序选择前N个候选线段
    相当于设置"候选线段池"的容量
生物学意义  ：
    值过低：可能遗漏真实存在的微管/微丝
    值过高：会增加计算量，可能引入噪声线段
推荐调整策略  ：
    细胞密度高（如癌细胞）→ 500-800
    正常细胞 → 300-500
    稀疏样本 → 200-300

2. hough_threshold: 0.3
功能  ：
    动态阈值系数，控制线段检测的灵敏度
工作机制  ：
    阈值 = max(当前层霍夫矩阵值) × 该系数
    仅保留得票数超过该阈值的候选线段
生物学意义  ：
    值过低（<0.2）：检测到更多断裂的短线段，适合高度破碎的病理样本
    值过高（>0.5）：仅保留强信号线段，适合信噪比低的荧光图像
黄金法则  ：
    若发现长线段被截断 → 降低阈值
    若出现大量杂乱短线 → 提高阈值

3. fill_gap: 8（像素）
功能  ：
    允许连接的线段端点最大间距
工作机制  ：
    在端点间距≤该值时，将相邻线段合并为单一长线段
生物学意义  ：
    值过小：无法修复因成像噪声导致的骨架断裂
    值过大：可能错误连接不同走向的纤维
结构指导  ：
    微管连续性好的样本 → 3-5px
    化疗后断裂样本 → 8-12px
    胶原纤维网络 → 5-8px
以下是霍夫变换相关超参数的详细功能解析（基于3D细胞骨架分析场景优化）：

4. min_line_length: 15（像素）
功能  ：
    有效线段的最小长度阈值
工作机制  ：
    过滤掉长度小于该值的候选线段
生物学意义  ：
    值过小：保留噪声产生的伪线段
    值过大：丢失真实存在的短纤维
细胞类型参考  ：
    微丝（>10μm） → ~30px（假设0.33μm/px）
    中间纤维 → 15-20px
    伪足 → 10-15px
换算公式  ：
    物理长度需求（μm） ÷ 像素分辨率（μm/px） = 像素值
'''

# 使用示例
if __name__ == "__main__":
    test_dir = r'E:\ZZD\Code\Cytoskeleton\experiment\A250523_M230925\subdata\r02c04'
    out_dir = r'E:\ZZD\Code\Cytoskeleton\experiment\A250523_M230925\subdata\skeleton_result\tmp'
    # out_dir = r'E:\ZZD\Code\Cytoskeleton\Cytoskeleton\Cytoskeleton_3D\output\idr_50_10\ridge_method\sato'

    all_paths = [f for f in os.listdir(test_dir) if f.endswith(('.tiff', '.tif'))]
    test_file = os.path.join(test_dir, all_paths[33])

    mask_path = r'E:\ZZD\Code\Cytoskeleton\experiment\A250523_M230925\subdata\cellprofiler_result\mask\B04_01_01_001--cell.png'
    img_info = Img_Information(test_file, output_dir=out_dir, maskpath=mask_path)
    img_info.change_dim_res('X', 0.185)
    img_info.change_dim_res('Y', 0.185)

    analyzer = CytoskeletonAnalyzer3D(img_info, intermediate=None, params=PROCESSING_PARAMS)
    analyzer.analyze_objects(debug_visualization=False)

    analyzer.generate_report(out_dir)



