import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl

mpl.use('Agg')
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from sklearn.decomposition import PCA

# ====================== 3. 分支特征计算 ======================
class BranchFeatureCalculator:
    """分支特征计算器"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_branch_features(self):
        """基于图论模型统一计算分支特征，并返回聚合特征和非聚合特征表格"""
        # 1. 构建映射关系
        self.extractor.build_mappings()

        # 2. 使用专用函数识别分支结构
        branches = self._identify_branches()

        # 3. 计算每个分支的特征
        branch_features_list = []  # 存储每个分支的特征字典

        # 计算形态特征
        morphology_data = self._calculate_branch_morphology(branches)
        # 计算拓扑特征
        topology_data = self._calculate_branch_topology(branches)
        # 计算空间分布特征
        spatial_data = self._calculate_branch_spatial(branches)
        # 计算强度特征
        intensity_data = self._calculate_branch_intensity(branches)

        # 为每个分支创建一个特征字典
        for i, branch in enumerate(branches):
            feat_dict = {
                'branch_id': i  # 使用简单数字ID
            }

            # 统一添加各类特征
            feature_sources = [
                ('形态', morphology_data),
                ('拓扑', topology_data),
                ('空间分布', spatial_data),
                ('强度', intensity_data)
            ]

            for category, data_dict in feature_sources:
                for key, values in data_dict.items():
                    # 使用_get_value安全获取值，避免索引错误
                    feat_dict[key] = self.extractor._get_value(values, i)

            branch_features_list.append(feat_dict)

        # 创建非聚合特征表格（DataFrame）- 不使用索引
        non_aggregated_df = pd.DataFrame(branch_features_list)

        return non_aggregated_df

    def _identify_branches(self):
        """识别分支结构（连通分量）"""
        branches = []

        # 如果图不存在，返回空列表
        if not self.extractor.graph:
            return branches

        # 获取连通分量
        components = list(nx.connected_components(self.extractor.graph))

        for branch_id, comp in enumerate(components):
            # 创建子图
            subgraph = self.extractor.graph.subgraph(comp)

            # 计算分支端点（度=1的节点）
            endpoints = [n for n, d in subgraph.degree() if d == 1]

            # 计算分支长度（所有边长度之和）
            total_length = 0
            segments = []  # 存储所有片段
            branch_pixels = []  # 存储分支的所有像素点（像素坐标）

            # 遍历所有边
            for u, v in subgraph.edges():
                # 尝试两种可能的边键顺序
                edge_key1 = (u, v)
                edge_key2 = (v, u)

                # 检查边属性是否存在
                if edge_key1 in self.extractor.edge_properties:
                    props_list = self.extractor.edge_properties[edge_key1]
                elif edge_key2 in self.extractor.edge_properties:
                    props_list = self.extractor.edge_properties[edge_key2]
                else:
                    continue

                for prop_idx, props in enumerate(props_list):
                    # 累加长度
                    seg_length = props.get('length', 0)
                    total_length += seg_length

                    # 创建线段对象
                    segment = {
                        'id': f"{u}_{v}_{prop_idx}" if edge_key1 in self.extractor.edge_properties else f"{v}_{u}_{prop_idx}",
                        'edge_key': edge_key1 if edge_key1 in self.extractor.edge_properties else edge_key2,
                        'props': props
                    }
                    segments.append(segment)

                    # 收集路径点（像素坐标）
                    if 'path' in props:
                        for point in props['path']:
                            branch_pixels.append(point)

            branches.append({
                'id': branch_id,
                'subgraph': subgraph,
                'endpoints': endpoints,
                'total_length': total_length,
                'segment_count': len(segments),
                'segments': segments,
                'pixels': branch_pixels  # 像素坐标
            })

        return branches


    def _calculate_branch_morphology(self, branches):
        """计算分支形态特征 - 基本物理属性"""
        morphology_data = {
            'length': [],  # 分支总长度（物理单位）；量化分支的尺寸
            # 'segment_count': [],  # 片段数量；量化分支的复杂性
            # 'tortuosity': [],  # 曲折度（总长度/端点间直线距离）；量化分支的弯曲程度
            'branching_factor': [],  # 分支因子（片段数/长度）；量化分支的密集程度
            'extent': [],  # 范围（边界框填充率）；量化分支的空间利用率
            'shape_complexity': [],  # 形状复杂度（熵）；量化分支形状的不规则性

            'aspect_ratio': [],  # 长宽比（主轴长度/最小轴长度）；量化分支的形状各向异性
            'axis_length_maj': [],  # 主轴长度；量化分支在主要方向上的延伸
            'axis_length_min': [],  # 最小轴长度；量化分支在最小方向上的延伸
            'shape_anisotropy': [],  # 形状各向异性（1-最小特征值/最大特征值）；量化分支的方向偏好性
        }
        # 3D特有特征
        if self.extractor.is_3d:
            morphology_data['axis_length_med'] = []  # 中轴长度（3D）；量化分支在次要方向上的延伸

        for branch in branches:
            # 基本形态特征
            # 精确累加所有线段物理长度
            branch_length_um = 0.0

            # 遍历分支所有线段
            for segment in branch['segments']:
                # 直接计算每个线段的物理长度
                seg_path = segment['props']['path']
                branch_length_um += self.extractor._calculate_physical_path_length(seg_path)

            morphology_data['length'].append(branch_length_um)

            # # 精确计算分支曲折度
            # if branch['endpoints'] and len(branch['endpoints']) >= 2:
            #     # 获取端点物理坐标
            #     start_id, end_id = branch['endpoints'][0], branch['endpoints'][1]
            #     start_coord = self.extractor.centroids[start_id]
            #     end_coord = self.extractor.centroids[end_id]
            #
            #     start_phy = self.extractor._to_physical_coordinates(start_coord)
            #     end_phy = self.extractor._to_physical_coordinates(end_coord)
            #
            #     # 计算端点物理直线距离
            #     linear_dist_um = np.linalg.norm(np.array(end_phy) - np.array(start_phy))
            #
            #     tortuosity = branch_length_um / linear_dist_um if linear_dist_um > 1e-6 else 1.0
            # else:
            #     tortuosity = np.nan
            #
            # morphology_data['tortuosity'].append(tortuosity)
            # # morphology_data['segment_count'].append(branch['segment_count'])


            # 分支因子（片段数/长度）
            if branch['total_length'] > 0:
                branching_factor = branch['segment_count'] / branch['total_length']
                morphology_data['branching_factor'].append(branching_factor)
            else:
                morphology_data['branching_factor'].append(np.nan)

            # 基于像素点的形态学特征计算
            if branch['pixels']:
                # 创建分支的二值图像，使用uint8类型避免警告
                branch_mask = np.zeros_like(self.extractor.skeleton, dtype=np.uint8)
                for pixel in branch['pixels']:
                    if self.extractor.is_3d:
                        # 确保坐标是整数
                        z, y, x = map(int, pixel)
                        branch_mask[z, y, x] = 1
                    else:
                        # 2D图像：确保坐标是整数
                        if len(pixel) == 3:  # (z,y,x)格式但2D
                            _, y, x = pixel
                        else:  # (y,x)格式
                            y, x = pixel
                        # 转换为整数
                        y = int(round(y))
                        x = int(round(x))
                        branch_mask[0, y, x] = 1

                # 计算区域属性
                labeled = measure.label(branch_mask)
                regions = measure.regionprops(labeled)

                if regions:
                    region = regions[0]  # 分支只有一个连通区域

                    # 范围（边界框填充率）
                    morphology_data['extent'].append(region.extent)

                    # 形状复杂度（熵）- 修复类型转换警告
                    if hasattr(region, 'image'):
                        # 显式转换为uint8类型，避免自动转换警告
                        flat_image = region.image.astype(np.uint8).flatten()
                        hist = np.histogram(flat_image, bins=2)[0]
                        hist = hist / hist.sum()
                        entropy = -np.sum(hist * np.log(hist + 1e-6))
                        morphology_data['shape_complexity'].append(entropy)

                else:
                    # 没有区域时填充NaN
                    morphology_data['extent'].append(np.nan)
                    morphology_data['shape_complexity'].append(np.nan)
            else:
                # 没有像素点时填充NaN
                morphology_data['extent'].append(np.nan)
                morphology_data['shape_complexity'].append(np.nan)

            # === 统一使用特征值方法计算轴长度 ===
            # 获取特征值并进行稳定性处理
            eigenvalues = region.inertia_tensor_eigvals
            eigenvalues = np.maximum(eigenvalues, 0)  # 确保非负
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # 确保不为0
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

            # 计算轴长度
            voxel_mean = np.mean(self.extractor.voxel_size)
            maj_len = 2 * np.sqrt(5 * sorted_eigenvalues[0]) * voxel_mean
            min_len = 2 * np.sqrt(5 * sorted_eigenvalues[-1]) * voxel_mean

            morphology_data['axis_length_maj'].append(maj_len)
            morphology_data['axis_length_min'].append(min_len)

            # 统一计算长宽比
            if min_len > 0:
                aspect_ratio = maj_len / min_len
            else:
                aspect_ratio = np.nan
            morphology_data['aspect_ratio'].append(aspect_ratio)

            # 3D特有特征计算
            if self.extractor.is_3d:
                med_len = 2 * np.sqrt(5 * sorted_eigenvalues[1]) * voxel_mean
                morphology_data['axis_length_med'].append(med_len)

            # 形状各向异性 = 1 - (最小特征值/最大特征值)
            if sorted_eigenvalues[0] > 0:
                shape_anisotropy = 1 - (sorted_eigenvalues[-1] / sorted_eigenvalues[0])
            else:
                shape_anisotropy = np.nan
            morphology_data['shape_anisotropy'].append(shape_anisotropy)

        return morphology_data

    def _calculate_branch_topology(self, branches):
        """计算分支拓扑特征 - 支持多重图"""
        topology_data = {
            'node_count': [],  # 节点数量；量化分支的规模
            'edge_count': [],  # 边数量；量化分支的连接复杂性
            'edge_density': [],  # 边密度（实际边数/最大可能边数）；量化分支的连接密度
            'global_efficiency': [],  # 全局效率；量化信息在分支中的传递效率（0-1）
            'open_node_ratio': [],  # 开放节点比例（度=1的节点比例）；量化分支的终端数量
            # 'node_compactness': [],  # 节点紧密度（平均距离/等效半径）；量化节点在分支中的聚集程度
            'avg_branch_angle': [],  # 分支点处相邻分支的平均角度；量化分支的分叉形态
        }

        for branch in branches:
            subgraph = branch['subgraph']
            n = subgraph.number_of_nodes()
            e = subgraph.number_of_edges()

            # 基本拓扑特征
            topology_data['node_count'].append(n)
            topology_data['edge_count'].append(e)

            # 边密度 = 实际边数 / 最大可能边数
            max_edges = n * (n - 1) / 2 if n > 1 else 0
            topology_data['edge_density'].append(e / max_edges if max_edges > 0 else 0)


            # 聚类系数和全局效率
            try:
                if n > 0:
                    # 全局效率 - 分支是连通分量，所以应该是连通的
                    if nx.is_connected(subgraph):
                        global_efficiency = nx.global_efficiency(subgraph)
                    else:
                        # 理论上分支应该是连通的，但安全起见
                        global_efficiency = np.nan
                    topology_data['global_efficiency'].append(global_efficiency)
                else:
                    topology_data['global_efficiency'].append(np.nan)
            except (nx.NetworkXError, ZeroDivisionError):
                topology_data['global_efficiency'].append(np.nan)

            # 计算开放节点比例
            degrees = [d for _, d in subgraph.degree()]
            open_node_count = sum(1 for d in degrees if d == 1)
            open_node_ratio = open_node_count / n if n > 0 else 0.0
            topology_data['open_node_ratio'].append(open_node_ratio)

            # # 计算节点紧密度
            # node_compactness = np.nan
            # if n > 0:
            #     # 获取分支中心（物理坐标）
            #     node_coords = []
            #     for node_id in subgraph.nodes():
            #         coord = np.array(self.extractor.centroids[node_id])
            #         node_coords.append(coord)
            #
            #     centroid_phy = np.mean(node_coords, axis=0)
            #
            #     # 计算分支等效半径（使用实际像素点）
            #     if self.extractor.is_3d:
            #         volume = len(branch['pixels']) * np.prod(self.extractor.voxel_size)
            #         equiv_radius = (3 * volume / (4 * np.pi)) ** (1/3)
            #     else:
            #         area = len(branch['pixels']) * np.prod(self.extractor.voxel_size)
            #         equiv_radius = np.sqrt(area / np.pi)
            #
            #     # 计算节点到中心的平均距离
            #     dists = []
            #     for coord in node_coords:
            #         dist = np.linalg.norm(coord - centroid_phy)
            #         dists.append(dist)
            #
            #     if dists and equiv_radius > 0:
            #         avg_dist = np.mean(dists)
            #         node_compactness = avg_dist / equiv_radius
            # topology_data['node_compactness'].append(node_compactness)

            # 计算分支点处的分支角度
            branch_angles = self._calculate_branch_angles(branch)
            topology_data['avg_branch_angle'].append(np.mean(branch_angles) if branch_angles else np.nan)

        return topology_data

    def _calculate_branch_spatial(self, branches):
        """计算分支空间分布特征（位置和方向）"""
        spatial_data = {
            'orientation_order': []  # 方向有序参数；量化分支内线段方向的一致性（0-1）
        }

        for branch in branches:
            # 初始化所有特征为NaN
            for key in spatial_data.keys():
                spatial_data[key].append(np.nan)

            # 跳过没有像素点的分支
            if not branch['pixels']:
                continue

            # 创建分支的二值图像
            branch_mask = np.zeros_like(self.extractor.skeleton, dtype=bool)
            for pixel in branch['pixels']:
                if self.extractor.is_3d:
                    # 确保坐标是整数
                    z, y, x = [int(round(coord)) for coord in pixel]
                    branch_mask[z, y, x] = True
                else:
                    if len(pixel) == 3:  # (z,y,x)格式但2D
                        _, y, x = pixel
                    else:  # (y,x)格式
                        y, x = pixel
                    # 转换为整数
                    y = int(round(y))
                    x = int(round(x))
                    branch_mask[0, y, x] = True

            # 计算区域属性
            labeled = measure.label(branch_mask)
            regions = measure.regionprops(labeled)

            if not regions:
                continue

            region = regions[0]  # 分支只有一个连通区域

            # === 计算分支方向特征 ===
            # 收集所有边的方向向量
            directions = []
            for segment in branch['segments']:
                path = segment['props'].get('path', [])
                if len(path) >= 2:
                    direction = self.extractor._calculate_filament_direction(path)
                    if np.linalg.norm(direction) > 1e-6:
                        directions.append(direction)

            if directions:
                directions = np.array(directions)

                # 计算方向有序参数
                oop_value = self.extractor._compute_oop(directions)
                spatial_data['orientation_order'][-1] = oop_value

        return spatial_data
    def _calculate_branch_intensity(self, branches):
        """计算分支强度特征 - 使用统一方法"""
        intensity_data = {
            'IntegratedIntensity': [],
            'MeanIntensity': [],
            'StdIntensity': [],
            'MaxIntensity': [],
            'MinIntensity': [],
            'MassDisplacement_um': [],
            'LowerQuartileIntensity': [],
            'MedianIntensity': [],
            'MADIntensity': [],
            'CVIntensity': [],
            'UpperQuartileIntensity': [],
            'Location_CenterMassIntensity_um': [],
            'Location_MaxIntensity_um': [],
            'SkewnessIntensity': [],
            'KurtosisIntensity': [],
            'capacity': [],  # 传输能力 = 平均强度 / 片段长度，量化单位长度细丝的荧光信号强度。
        }

        for branch in branches:
            points = branch['pixels']

            # 使用统一方法计算强度特征
            features = self.extractor._compute_unified_intensity_features(points)

            # 存储所有12个特征
            for key in intensity_data.keys():
                if key != 'capacity':  # capacity需要单独计算
                    intensity_data[key].append(features.get(key, np.nan))

            # 计算分支容量（总强度/总长度）
            total_intensity = features['IntegratedIntensity']
            total_length = branch['total_length']
            capacity = total_intensity / max(total_length, 1e-6)
            intensity_data['capacity'].append(capacity)

            # 存储完整特征集到分支属性
            branch['intensity_features'] = features

        return intensity_data

    def _calculate_branch_angles(self, branch):
        """计算分支点处的分支角度"""
        branch_angles = []
        subgraph = branch['subgraph']

        for node in subgraph.nodes():
            # 只处理分支点（度≥3的节点）
            if subgraph.degree(node) < 3:
                continue

            # 获取节点体素坐标
            node_coord = np.array(self.extractor.centroids[node])

            # 计算分支点到每个邻居节点的单位向量
            vectors = []
            for neighbor in subgraph.neighbors(node):
                n_coord = np.array(self.extractor.centroids[neighbor])
                vec = n_coord - node_coord
                norm_val = np.linalg.norm(vec)
                if norm_val > 1e-6:
                    vectors.append(vec / norm_val)

            # 计算所有相邻分支对的夹角
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    dot = np.clip(np.dot(vectors[i], vectors[j]), -1, 1)
                    angle = np.arccos(dot)  # 弧度
                    branch_angles.append(np.degrees(angle))  # 转换为度数

        return branch_angles
