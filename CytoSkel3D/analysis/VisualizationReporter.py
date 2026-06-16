import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import io, measure, exposure
import traceback

class VisualizationReporter:
    """处理网络可视化与报告生成"""

    def __init__(self, self_info):
        # 存储对主分析器实例的引用
        self.analyzer = self_info.analyzer
        self.info = self_info

    def visualize_3d_paths(self, output_dir, obj_id):
        """绘制完整的3D网络路径"""
        if not self.analyzer.is_3d or len(self.analyzer.edge_properties) == 0:
            return

        try:
            # 创建3D图形
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # 设置背景色
            fig.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')

            base_node_size = 15

            # 收集所有边的容量值用于创建全局颜色映射
            capacities = []
            for (u, v), properties_list in self.analyzer.edge_properties.items():
                for props in properties_list:
                    if 'path' in props and 'capacity' in props:
                        capacity = props['capacity']
                        if np.isfinite(capacity) and capacity > 0:
                            capacities.append(capacity)

            # 创建颜色映射（使用所有边的数据归一化）
            if capacities:
                # 使用实际数据范围而不是百分位数
                vmin = min(capacities)
                vmax = max(capacities)

                # 如果数据范围太小，适当扩展范围以确保颜色变化可见
                if vmax - vmin < 1e-5:  # 所有值几乎相同
                    vmin = vmin * 0.9 if vmin > 0 else 0
                    vmax = vmax * 1.1 if vmax > 0 else 1.0

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap('viridis')
                mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                mapper.set_array([])  # 必须设置空数组以供颜色条使用
            else:
                print("警告：没有有效的传输能力值可用于颜色映射")
                mapper = None

            # 绘制所有边路径
            for (u, v), properties_list in self.analyzer.edge_properties.items():
                for props in properties_list:
                    if 'path' not in props:
                        continue

                    # 获取路径并转换为像素坐标
                    path = props['path']
                    path_pixels = self._physical_to_pixel_coordinates(path)

                    # 确保有有效的坐标点
                    if not path_pixels:
                        continue

                    zs, ys, xs = zip(*[(p[0], p[1], p[2]) for p in path_pixels])

                    # 计算颜色
                    if mapper and 'capacity' in props:
                        capacity = props['capacity']
                        # 限制在有效范围内
                        if capacity < vmin: capacity = vmin
                        if capacity > vmax: capacity = vmax
                        color = mapper.to_rgba(capacity)
                    else:
                        color = 'cyan'  # 默认颜色

                    # 绘制路径
                    ax.plot(xs, ys, zs,
                            color=color,
                            linewidth=2,  # 固定线宽为2
                            alpha=0.7,
                            solid_capstyle='round'
                            )


            # 收集所有节点位置并统一绘制（保持不变）
            xs_n, ys_n, zs_n = [], [], []
            node_sizes = []
            for node_id, node_data in self.analyzer.graph.nodes(data=True):
                if 'pos' in node_data:
                    pixel_coord = self._physical_to_pixel_coordinates([node_data['pos']])
                    if pixel_coord:
                        z, y, x = pixel_coord[0]
                        xs_n.append(x)
                        ys_n.append(y)
                        zs_n.append(z)
                        degree = self.analyzer.graph.degree[node_id]
                        size = max(5, min(30, base_node_size + degree * 1.5))
                        node_sizes.append(size)

            # 统一绘制所有节点（保持不变）
            if xs_n and ys_n and zs_n and node_sizes:
                ax.scatter(xs_n, ys_n, zs_n,
                           s=node_sizes,
                           c='lime',
                           ec='white',
                           alpha=0.7)

            # 设置坐标轴标签（保持不变）
            unit_x = "X (μm)" if self.analyzer.voxel_size[2] != 1 else "X (px)"
            unit_y = "Y (μm)" if self.analyzer.voxel_size[1] != 1 else "Y (px)"
            unit_z = "Z (μm)" if self.analyzer.voxel_size[0] != 1 else "Z (px)"

            ax.set_xlabel(unit_x)
            ax.set_ylabel(unit_y)
            ax.set_zlabel(unit_z)

            # 设置视角
            ax.view_init(elev=30, azim=45)

            # 标题
            ax.set_title(f"Object {obj_id} 3D Network Paths", color='white', fontsize=14)

            # 保存图像
            output_path = os.path.join(output_dir, f"object_{obj_id}_3d_paths.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"3D路径可视化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    def visualize_network_projection(self, output_dir, obj_id, bg_img=None):
        """在2D投影上绘制网络路径（XY投影）"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # 设置背景
        if bg_img is not None:
            ax.imshow(bg_img, cmap='gray', origin='upper')
        else:
            if self.analyzer.is_3d:
                bg_img = np.max(self.analyzer.skeleton, axis=0)
            else:
                bg_img = self.analyzer.skeleton
            ax.imshow(bg_img, cmap='gray', origin='upper')

        # 定义合理的节点大小
        base_node_size = 10

        # 收集所有边的容量值用于创建全局颜色映射
        capacities = []
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' in props and 'capacity' in props:
                    capacity = props['capacity']
                    if np.isfinite(capacity) and capacity > 0:
                        capacities.append(capacity)

        # 创建颜色映射（使用所有边的数据归一化）
        if capacities:
            # 使用实际数据范围而不是百分位数
            vmin = min(capacities)
            vmax = max(capacities)

            # 如果数据范围太小，适当扩展范围以确保颜色变化可见
            if vmax - vmin < 1e-5:  # 所有值几乎相同
                vmin = vmin * 0.9 if vmin > 0 else 0
                vmax = vmax * 1.1 if vmax > 0 else 1.0

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('viridis')
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])  # 必须设置空数组以供颜色条使用
        else:
            print("警告：没有有效的传输能力值可用于颜色映射")
            mapper = None

        # 绘制所有边路径
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' not in props:
                    continue

                path = props['path']
                xs, ys = [], []

                # 将路径从物理坐标转换为像素坐标
                path_pixels = self._physical_to_pixel_coordinates(path)

                # 确保有有效的坐标点
                if not path_pixels:
                    continue

                for point in path_pixels:
                    if self.analyzer.is_3d:
                        z, y, x = point
                    else:
                        if len(point) == 3:  # (z, y, x)
                            z, y, x = point
                        else:  # (y, x)
                            y, x = point
                    xs.append(x)
                    ys.append(y)

                # 确保有足够的数据点
                if len(xs) < 2:
                    continue

                # 计算颜色
                if mapper and 'capacity' in props:
                    capacity = props['capacity']
                    # 限制在有效范围内
                    if capacity < vmin: capacity = vmin
                    if capacity > vmax: capacity = vmax
                    color = mapper.to_rgba(capacity)
                else:
                    color = 'cyan'  # 默认颜色

                # 绘制路径
                ax.plot(xs, ys,
                        color=color,
                        linewidth=2,
                        alpha=0.7,
                        solid_capstyle='round'
                        )

        # 收集所有节点位置和大小
        xs, ys = [], []
        node_sizes = []
        for node_id, node_data in self.analyzer.graph.nodes(data=True):
            if 'pos' in node_data:
                pixel_coords = self._physical_to_pixel_coordinates([node_data['pos']])
                if pixel_coords:
                    coord = pixel_coords[0]
                    if self.analyzer.is_3d and len(coord) == 3:
                        z, y, x = coord
                    elif len(coord) >= 2:
                        x, y = coord[:2][::-1]
                    else:
                        continue

                    xs.append(x)
                    ys.append(y)

                    degree = self.analyzer.graph.degree[node_id]
                    size = max(5, min(30, base_node_size + degree * 2))
                    node_sizes.append(size)

        # 统一绘制所有节点
        if xs and ys and node_sizes:
            ax.scatter(
                xs, ys,
                s=node_sizes,
                c='lime',
                ec='white',
                alpha=0.7,
                zorder=5
            )

        # 设置标签
        unit = " (px)"
        if any(v != 1 for v in self.analyzer.voxel_size):
            unit = " (μm)"

        ax.set_xlabel(f"X{unit}")
        ax.set_ylabel(f"Y{unit}")
        ax.set_title(f"Object {obj_id} Network Projection", fontsize=12)

        # 保存图像
        output_path = os.path.join(output_dir, f"object_{obj_id}_network_projection.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _physical_to_pixel_coordinates(self, physical_points):
        voxel_size = self.analyzer.voxel_size
        pixel_points = []
        for point in physical_points:
            if len(point) == 3:
                z, y, x = point
                pixel_points.append((
                    z / voxel_size[0] if voxel_size[0] > 0 else 0,
                    y / voxel_size[1] if voxel_size[1] > 0 else 0,
                    x / voxel_size[2] if voxel_size[2] > 0 else 0
                ))
            elif len(point) == 2:
                y, x = point
                pixel_points.append((
                    0,
                    y / voxel_size[1] if voxel_size[1] > 0 else 0,
                    x / voxel_size[2] if voxel_size[2] > 0 else 0
                ))
        return pixel_points

    def export_paths_as_tiff(self, output_path, obj_id):
        """将重建的路径输出为3D TIFF图像"""
        if not self.analyzer.is_3d: return
        path_image = np.zeros_like(self.analyzer.skeleton, dtype=np.uint16)
        path_id = 1
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' in props:
                    for point in props['path']:
                        z, y, x = map(int, point[:3])
                        if 0 <= z < path_image.shape[0] and 0 <= y < path_image.shape[1] and 0 <= x < path_image.shape[2]:
                            path_image[z, y, x] = path_id
                path_id += 1
        save_path = os.path.join(output_path, f"object_{obj_id}_reconstructed_paths.tiff")
        io.imsave(save_path, path_image, check_contrast=False)
        print(f"路径重建图像已保存至: {save_path}")

    # =========================================================================
    #  [新] 网络构建可视化 (从 Analyzer 迁移并重构)
    # =========================================================================

    def visualize_network_construction(self, segments_bp, segments_connected):
        """
        可视化网络构建的关键步骤
        参数来自 analyzer.analyze_network 的返回值
        """
        # 重新计算临时连通域用于可视化 (保持Analyzer纯净，不存储非核心状态)
        connectivity = 3 if self.analyzer.is_3d else 2
        labeled = measure.label(self.analyzer.skeleton, connectivity=connectivity)

        if self.analyzer.full_image_mode:
            object_visualize_path = self.analyzer.img_info.graph_dir
        else:
            object_visualize_path = os.path.join(self.analyzer.img_info.graph_dir,
                                                 "object_" + str(self.analyzer.object_id))

        os.makedirs(object_visualize_path, exist_ok=True)

        # 1. 可视化网络图结构 (Node-Edge Graph)
        graph_output_path = os.path.join(object_visualize_path, "network_graph.png")
        self.visualize_network_graph(output_path=graph_output_path)


    def visualize_network_graph(self, output_path=None):
        """可视化网络图结构 (NetworkX Graph)"""
        if not self.analyzer.graph:
            print("网络图尚未构建，无法可视化")
            return

        plt.figure(figsize=(12, 10))

        if self.analyzer.is_3d:
            self._visualize_3d_network_graph()
        else:
            self._visualize_2d_network_graph()

        plt.title("Cytoskeleton Network Graph")
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # --- 内部辅助绘图方法 ---

    def _visualize_2d_network_graph(self):
        graph = self.analyzer.graph
        pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
        node_colors = self._get_node_colors(graph)

        nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=node_colors, alpha=0.8)
        self._draw_graph_edges(graph, pos)
        self._add_graph_legend()

    def _visualize_3d_network_graph(self):
        # 3D投影到2D平面可视化
        self._visualize_2d_network_graph()

    def _get_node_colors(self, graph):
        colors = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            if node_type == 'endpoint':
                colors.append('red')
            elif node_type == 'branchpoint':
                colors.append('blue')
            elif node_type == 'super_branchpoint':
                colors.append('purple')
            else:
                colors.append('gray')
        return colors

    def _draw_graph_edges(self, graph, pos):
        for edge in graph.edges():
            edge_data = graph.get_edge_data(*edge)
            for key, data in edge_data.items():
                linewidth = 1 + 2 * min(data.get('bending', 1.0), 3.0)
                capacity = data.get('capacity', 0)

                if capacity > 0.5:
                    edge_color = 'green'
                elif capacity > 0.2:
                    edge_color = 'orange'
                else:
                    edge_color = 'gray'

                nx.draw_networkx_edges(
                    graph, pos, edgelist=[edge], width=linewidth,
                    edge_color=edge_color, alpha=0.6, arrows=False
                )

    def _add_graph_legend(self):
        plt.scatter([], [], c='red', s=50, label='Endpoints')
        plt.scatter([], [], c='blue', s=50, label='Branchpoints')
        plt.scatter([], [], c='purple', s=50, label='Super Branchpoints')
        plt.plot([], [], color='green', linewidth=2, label='High Capacity')
        plt.plot([], [], color='orange', linewidth=2, label='Medium Capacity')
        plt.plot([], [], color='gray', linewidth=2, label='Low Capacity')
        plt.legend(loc='best')

    def _visualize_segments(self, ax, segments, title, max_label):
        shape = self.analyzer.skeleton.shape[1:] if self.analyzer.is_3d else self.analyzer.skeleton.shape[1:]
        vis_img = np.zeros(shape, dtype=np.uint16)

        for i, seg in enumerate(segments):
            path = seg['path']
            coords = [(p[1], p[2]) for p in path]  # 假设3D或2D都在最后两维
            for y, x in coords:
                if 0 <= y < vis_img.shape[0] and 0 <= x < vis_img.shape[1]:
                    vis_img[y, x] = i + 1

        num_labels = max_label + 1
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_labels))
        colors[0] = [0, 0, 0, 1]

        ax.imshow(vis_img, cmap=mpl.colors.ListedColormap(colors))
        ax.set_title(title)
        ax.axis('off')

    def _visualize_labeled_skeleton(self, ax, labeled_skeleton, title):
        if self.analyzer.is_3d:
            vis_img = np.max(labeled_skeleton, axis=0)
        else:
            vis_img = labeled_skeleton[0] if labeled_skeleton.ndim == 3 else labeled_skeleton

        num_labels = np.max(vis_img) + 1
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_labels))
        colors[0] = [0, 0, 0, 1]

        ax.imshow(vis_img, cmap=mpl.colors.ListedColormap(colors))
        ax.set_title(title)
        ax.axis('off')

    def _visualize_pixel_class(self, ax, title):
        pixel_class = self.analyzer.pixel_class
        if self.analyzer.is_3d:
            vis_img = np.max(pixel_class, axis=0)
        else:
            vis_img = pixel_class[0] if pixel_class.ndim == 3 else pixel_class

        color_img = np.zeros((*vis_img.shape, 3), dtype=np.uint8)
        color_img[vis_img == 1] = [255, 0, 0]  # Endpoints
        color_img[vis_img == 2] = [0, 255, 0]  # Lines
        color_img[vis_img == 3] = [0, 0, 255]  # Branchpoints
        color_img[vis_img == 4] = [128, 0, 128]  # Super Nodes

        ax.imshow(color_img)
        ax.set_title(title)
        ax.axis('off')

        coords = np.argwhere(vis_img > 0)
        for y, x in coords:
            val = vis_img[y, x]
            color = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple'}.get(val, 'white')
            if val in [2, 4]:
                ax.scatter(x, y, s=50, c=color, alpha=0.9, linewidths=2)


class VisualizationManager:
    """
    细胞骨架可视化管理器

    参数:
    analyzer (CytoskeletonAnalyzer3D): 细胞骨架分析器实例
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.img_info = analyzer.img_info
        self.is_3d = analyzer.is_3d
        self.voxel_size = analyzer.voxel_size
        self.network_cache = analyzer.network_cache
        self.feature_table = analyzer.feature_table

    def visualize_global_network(self, edge_attribute='length', cmap='viridis'):
        """绘制所有对象的整体网络投影 - 优化版（使用edge_properties）"""
        # 创建2行1列的图形布局
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 24))

        # 设置图形和子图2的背景为黑色
        fig.patch.set_facecolor('black')
        ax2.set_facecolor('black')

        # 获取背景图像
        if self.is_3d:
            raw_bg = np.max(self.analyzer.raw_image, axis=0)
        else:
            raw_bg = self.analyzer.raw_image

        # 对原始图像进行max-min归一化
        raw_min = np.min(raw_bg)
        raw_max = np.max(raw_bg)
        if raw_max > raw_min:
            raw_bg = (raw_bg - raw_min) / (raw_max - raw_min)
        else:
            raw_bg = exposure.rescale_intensity(raw_bg, out_range=(0, 1))

        # 子图1：显示归一化后的原始图像（无前景）
        im1 = ax1.imshow(raw_bg, cmap='gray', origin='upper', vmin=0, vmax=1)
        ax1.set_title("Normalized Original Image", fontsize=25, color='white')
        ax1.set_xlabel("X (μm)" if self.analyzer.apply_anisotropic else "X (px)")
        ax1.set_ylabel("Y (μm)" if self.analyzer.apply_anisotropic else "Y (px)")
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        # 准备颜色映射数据
        all_values = []  # 收集所有边的属性值

        # 第一遍：收集所有边的属性值
        for obj_id, analyzer in self.network_cache.items():
            if not hasattr(analyzer, 'edge_properties'):
                continue

            for edge_key, properties_list in analyzer.edge_properties.items():
                for props in properties_list:
                    value = props.get(edge_attribute, 0)
                    all_values.append(value)

        # 增强的颜色映射处理
        if all_values:
            # 使用百分位数定义范围，避免异常值影响
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)

            # 确保vmax > vmin
            if vmax <= vmin:
                # 当所有值几乎相同时的处理
                if vmin > 0:
                    vmax = vmin * 1.1
                else:
                    vmin, vmax = 1e-6, 1.0

            # 创建归一化器和映射器
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])
        else:
            print("警告：没有有效的传输能力值可用于颜色映射")
            norm = None
            mapper = None

        # 第二遍：绘制路径（只在子图2绘制）
        for obj_id, analyzer in self.network_cache.items():
            if not hasattr(analyzer, 'edge_properties'):
                continue

            # 遍历所有边及其片段
            for edge_key, properties_list in analyzer.edge_properties.items():
                for props in properties_list:
                    # 跳过没有路径数据的边
                    if 'path' not in props or len(props['path']) < 2:
                        continue

                    # 获取属性值
                    value = props.get(edge_attribute, 0)
                    path = props['path']

                    # 准备坐标列表
                    xs, ys = [], []

                    # 处理路径点
                    for point in path:
                        if self.is_3d:
                            # 3D点: (z,y,x) -> 投影到(y,x)
                            _, y, x = point[:3]
                        else:
                            # 2D点: 可能是(y,x)或(z,y,x)格式
                            if len(point) == 3:  # (z,y,x)格式但z=0
                                _, y, x = point
                            else:  # (y,x)格式
                                y, x = point
                        xs.append(x)
                        ys.append(y)

                    # 确保有足够的数据点
                    if len(xs) < 2:
                        continue

                    # 计算颜色
                    if mapper:
                        color = mapper.to_rgba(value)
                        # 固定线宽为1.0（不再根据值动态调整）
                        linewidth = 1.0
                    else:
                        color = 'cyan'
                        linewidth = 1.0

                    # 在子图2绘制网络路径
                    ax2.plot(xs, ys,
                             color=color,
                             linewidth=linewidth,
                             alpha=0.7,
                             solid_capstyle='round')

        # 设置子图2的标签和标题
        ax2.set_title("Global Network Projection", fontsize=25, color='white')
        ax2.set_xlabel("X (μm)" if self.analyzer.apply_anisotropic else "X (px)", color='white')
        ax2.set_ylabel("Y (μm)" if self.analyzer.apply_anisotropic else "Y (px)", color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')

        # 统一两个子图的比例和范围
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_aspect(ax1.get_aspect())

        # # 只翻转子图1的Y轴
        # ax1.invert_yaxis()

        # 修复：为子图2单独添加颜色条
        if mapper:
            # 获取子图2的位置信息
            ax2_pos = ax2.get_position()

            # 在子图2的右侧创建颜色条轴
            cax = fig.add_axes([ax2_pos.x1 + 0.02, ax2_pos.y0, 0.02, ax2_pos.height])
            cbar = fig.colorbar(mapper, cax=cax, orientation='vertical')
            cbar.set_label(edge_attribute.upper(), fontsize=12, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # 调整布局，确保颜色条不会重叠
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.img_info.graph_dir, "global_network_projection.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig)


    def visualize_global_3d_paths(self, edge_attribute='capacity'):
        """绘制所有对象的整体3D网络路径 - 修正版，基于capacity着色"""
        if not self.is_3d:
            return

        try:
            fig = plt.figure(figsize=(18, 16))
            ax = fig.add_subplot(111, projection='3d')

            # 收集所有边的capacity值
            all_capacities = []
            for obj_id, analyzer in self.network_cache.items():
                if not hasattr(analyzer, 'edge_properties'):
                    continue
                for edge_key, properties_list in analyzer.edge_properties.items():
                    for props in properties_list:
                        if 'path' not in props:
                            continue
                        capacity = props.get('capacity', 0)
                        all_capacities.append(capacity)

            # 处理空数据集
            if not all_capacities:
                print("警告：没有检测到有效的传输能力值")
                return

            # 确定归一化范围（使用百分位数避免异常值影响）
            valid_capacities = [c for c in all_capacities if c > 0]
            if not valid_capacities:
                print("警告：所有边的传输能力值为0或负数")
                return

            vmin = np.percentile(valid_capacities, 5)
            vmax = np.percentile(valid_capacities, 95)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])

            # 遍历所有对象和路径
            for obj_id, analyzer in self.network_cache.items():
                if not hasattr(analyzer, 'edge_properties'):
                    continue

                # 遍历所有边及其片段
                for edge_key, properties_list in analyzer.edge_properties.items():
                    for props in properties_list:
                        if 'path' not in props:
                            continue

                        path = props['path']
                        capacity = props.get(edge_attribute, 0)
                        color = mapper.to_rgba(capacity)  # 使用当前capacity值获取颜色

                        # 提取坐标
                        xs, ys, zs = [], [], []
                        for point in path:
                            if len(point) >= 3:  # 确保是3D点
                                z, y, x = point[:3]
                                xs.append(x)
                                ys.append(y)
                                zs.append(z)

                        if xs and ys and zs:  # 确保有有效的点
                            # 绘制路径
                            ax.plot(xs, ys, zs,
                                    color=color,
                                    linewidth=0.8 + 3 * (capacity / vmax),
                                    alpha=0.7,
                                    solid_capstyle='round')

            # 添加颜色条
            cbar = fig.colorbar(mapper, ax=ax, shrink=0.8, pad=0.1)
            cbar.set_label('Transport Capacity', fontsize=12)
            cbar.formatter = ScalarFormatter(useMathText=False)
            cbar.update_ticks()

            # 设置坐标轴
            ax.set_xlabel("X (μm)" if self.voxel_size[2] != 1 else "X (px)")
            ax.set_ylabel("Y (μm)" if self.voxel_size[1] != 1 else "Y (px)")
            ax.set_zlabel("Z (μm)" if self.voxel_size[0] != 1 else "Z (px)")

            # 设置视角
            ax.view_init(elev=25, azim=35)

            # 标题
            ax.set_title("Global 3D Network Paths (Capacity Colored)", fontsize=16)

            # 保存图像
            save_path = os.path.join(self.img_info.graph_dir, "global_3d_network_paths.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"全局3D网络可视化失败: {str(e)}")
            traceback.print_exc()
            plt.close('all')

    def visualize_network(self,
                          edge_attribute='capacity',
                          background_type='raw',
                          cmap='viridis',  # 保留颜色参数
                          output_suffix=None):
        """网络可视化方法（改进版）

        Args:
            edge_attribute: 用于着色的边属性
            background_type: 背景类型(raw/annotated)
            cmap: 颜色映射表（默认viridis）
            output_suffix: 输出文件名后缀
        """
        # 获取并预处理背景
        bg_img = self._get_background_image(background_type)
        bg_img = (bg_img - bg_img.min()) / (bg_img.max() - bg_img.min())
        # bg_img = exposure.rescale_intensity(bg_img, out_range=(0.1, 0.9))  # 智能对比度调整

        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(bg_img, cmap='gray', vmin=0.1, vmax=0.9)

        # 边数据收集（带校验）
        all_edges = []
        for obj in self.feature_table:
            # 安全获取对象ID
            obj_id = obj.get('object_id')
            if not obj_id:
                continue

            # 校验分析器有效性
            analyzer = self.network_cache.get(obj_id)
            if not analyzer or not hasattr(analyzer, 'graph'):
                continue

            # 生成节点坐标（过滤无效值）
            node_pos = {}
            for n, coord in enumerate(analyzer.centroids):
                try:
                    if self.is_3d:  # 3D图像处理
                        z, y, x = map(float, coord)
                        if all(np.isfinite([z, y, x])):
                            node_pos[n] = (
                                y * self.voxel_size[1],
                                x * self.voxel_size[2]
                            )
                    else:  # 2D图像处理
                        # 处理可能存在的(z,y,x)或(y,x)两种格式
                        if len(coord) == 3:  # (z,y,x)格式但z=0
                            _, y, x = coord
                        else:  # (y,x)格式
                            y, x = coord

                        if all(np.isfinite([y, x])):
                            node_pos[n] = (
                                y * self.voxel_size[1],
                                x * self.voxel_size[2]
                            )
                except (TypeError, ValueError) as e:
                    print(f"节点{n}坐标处理异常: {str(e)}")
                    continue

            # 收集边数据
            for u, v, data in analyzer.graph.edges(data=True):
                if u in node_pos and v in node_pos:
                    y0, x0 = node_pos[u]
                    y1, x1 = node_pos[v]
                    all_edges.append((
                        y0, x0, y1, x1,
                        data.get(edge_attribute, 0)
                    ))

        # 可视化边（带归一化）
        if all_edges:
            values = [e[4] for e in all_edges]
            valid_values = [v for v in values if v > 0]

            if valid_values:
                # 动态范围计算
                vmin = np.percentile(valid_values, 5)
                vmax = np.percentile(valid_values, 95)
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

                # 绘制线段
                for y0, x0, y1, x1, val in all_edges:
                    if val <= 0:
                        continue
                    ax.plot(
                        [x0, x1], [y0, y1],  # 注意XY坐标转换
                        color=mapper.to_rgba(val),
                        linewidth=0.3 + 2 * (val / vmax),  # 动态线宽
                        alpha=0.7,
                        solid_capstyle='round'
                    )

                # 添加颜色条
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(mapper, cax=cax, label=edge_attribute.upper())

        # 输出配置
        filename = f"network_{background_type}"
        if output_suffix:
            filename += f"_{output_suffix}"
        save_path = os.path.join(self.img_info.graph_dir, f"{filename}.png")

        ax.set_title(f"Integrated Network ({background_type.title()})")
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_paths(self,
                        background_type='raw',
                        edge_attribute='capacity',
                        cmap='viridis',
                        output_suffix=None):
        """可视化完整边路径（而非端点连线） - 新功能

        参数:
            background_type: 背景类型(raw/ridge/skeleton)
            edge_attribute: 用于着色的边属性(如'capacity'/'length')
            cmap: 颜色映射表
            output_suffix: 输出文件名后缀
        """
        # 获取并预处理背景
        bg_img = self._get_background_image(background_type)
        bg_img = exposure.rescale_intensity(bg_img, out_range=(0.1, 0.9))

        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(bg_img, cmap='gray', vmin=0.1, vmax=0.9)

        # 收集所有边的路径和属性值
        all_paths = []
        edge_values = []

        # 遍历所有对象和网络分析器
        for obj_id, analyzer in self.network_cache.items():
            # 检查分析器是否包含路径信息
            if not hasattr(analyzer, 'edge_properties'):
                continue

            # 遍历所有边路径
            for edge_key, props in analyzer.edge_properties.items():
                # 跳过没有路径数据的边
                if 'path' not in props or len(props['path']) < 2:
                    continue

                # 获取属性值
                value = props.get(edge_attribute, 0)
                if edge_attribute not in props:
                    # 尝试从图边获取属性
                    try:
                        graph_edge = analyzer.graph.edges[edge_key]
                        value = graph_edge.get(edge_attribute, 0)
                    except:
                        value = 0

                edge_values.append(value)
                all_paths.append(props['path'])

        # 如果没有有效路径，直接保存并返回
        if not all_paths:
            ax.set_title("No Paths Found")
            ax.axis('off')
            save_path = os.path.join(self.img_info.graph_dir, f"no_paths_warning.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 归一化属性值用于颜色映射
        valid_values = [v for v in edge_values if v > 0]
        if valid_values:
            vmin = np.percentile(valid_values, 5)
            vmax = np.percentile(valid_values, 95)
        else:
            vmin, vmax = 0, 1

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # 逐条路径绘制
        for path_idx, path_points in enumerate(all_paths):
            value = edge_values[path_idx]
            color = mapper.to_rgba(value) if valid_values else 'cyan'

            # 逐段绘制路径
            for i in range(1, len(path_points)):
                start = path_points[i - 1]
                end = path_points[i]

                # 处理不同维度的坐标
                if len(start) == 3:  # 3D点:(z,y,x)
                    y0, x0 = start[1], start[2]
                else:  # 2D点:(y,x)
                    y0, x0 = start[0], start[1] if len(start) > 1 else (0, 0)

                if len(end) == 3:  # 3D点:(z,y,x)
                    y1, x1 = end[1], end[2]
                else:  # 2D点:(y,x)
                    y1, x1 = end[0], end[1] if len(end) > 1 else (0, 0)

                ax.plot([x0, x1], [y0, y1],  # XY顺序转换
                        color=color,
                        linewidth=0.3 + 2 * (value / vmax if vmax > 0 else 0.3),
                        alpha=0.7,
                        solid_capstyle='round')

        # 添加颜色条（如果存在有效属性值）
        if valid_values:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(mapper, cax=cax, label=edge_attribute.upper())

        # 输出配置
        filename = f"paths_{background_type}"
        if output_suffix:
            filename += f"_{output_suffix}"
        save_path = os.path.join(self.img_info.graph_dir, f"{filename}.png")

        ax.set_title(f"Full Path Visualization ({edge_attribute})")
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _get_background_image(self, bg_type):
        """获取背景图像"""
        if self.is_3d:
            # 3D使用最大投影
            if bg_type == 'raw':
                bg_img = np.max(self.analyzer.raw_image, axis=0)
            elif bg_type == 'ridge':
                bg_img = np.max(self.analyzer.ridge_contrast_image, axis=0)
            elif bg_type == 'skeleton':
                bg_img = np.max(self.analyzer.skeleton, axis=0)
        else:
            # 2D直接使用原始图像
            if bg_type == 'raw':
                bg_img = self.analyzer.raw_image
            elif bg_type == 'ridge':
                bg_img = self.analyzer.ridge_contrast_image
            elif bg_type == 'skeleton':
                bg_img = self.analyzer.skeleton
        return bg_img

    def _select_cmap(self, bg_img):
        """智能选择背景颜色映射"""
        return 'gray_r' if np.median(bg_img) < 128 else 'gray'


