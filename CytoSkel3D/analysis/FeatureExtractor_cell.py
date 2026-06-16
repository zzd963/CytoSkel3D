import os
import numpy as np
import pandas as pd
import mahotas as mh
from warnings import warn
from scipy import ndimage, stats
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from sklearn.decomposition import PCA


# ====================== 5. 细胞特征计算 ======================
class CellFeatureCalculator:
    """细胞特征计算器（独立于网络特征）"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_cell_features(self):
        """计算细胞级别特征 - 独立于网络特征"""
        # 创建细胞对象
        cell = {
            'obj_mask': self.extractor.obj_mask,
            'raw_image': self.extractor.raw_image,
            'ridge_image': self.extractor.ridge_image,
        }

        # 计算细胞特征
        morphology_data = self._calculate_cell_morphology(cell)
        spatial_data = self._calculate_cell_spatial(cell)
        intensity_data = self._calculate_cell_intensity(cell)


        # 合并所有特征
        cell_features = {}
        cell_features.update(morphology_data)
        cell_features.update(intensity_data)
        cell_features.update(spatial_data)

        return pd.DataFrame([cell_features])

    def _calculate_cell_morphology(self, cell):
        """计算细胞形态特征 - 根据维度初始化不同的特征字典"""
        # 根据维度初始化不同的特征字典
        if self.extractor.is_3d:
            morphology_data = {
                # 3D形态特征
                'volume_um3': np.nan,  # 细胞体积（3D）；量化细胞的空间范围
                'convex_volume_um3': np.nan,  # 细胞凸包体积（3D）；量化细胞的空间范围
                'surface_um2': np.nan,  # 细胞表面积（3D）；量化细胞的表面复杂度
                'compactness': np.nan,  # 细胞紧实度（3D）；量化细胞的整体紧凑性
                'convex_density': np.nan,  # 细胞凸包密度（细胞体积/凸包体积）；量化细胞在凸包内的填充程度
                'max_diameter_um': np.nan,  # 细胞最大直径；量化细胞的最大延伸距离
                'med_diameter_um': np.nan,  # 中间直径（3D特有）:量化细胞的中间延伸距离
                'min_diameter_um': np.nan,  # 细胞最小直径；量化细胞的最小延伸距离
                'stretch': np.nan,  # 细胞伸展度（最大特征值-最小特征值）/最大特征值；量化细胞的形状各向异性
                'oblateness': np.nan,  # 细胞扁平度（3D）；量化细胞在垂直方向的扁平程度
                'aspect_ratio': np.nan,  # 长宽比（主轴长度/最小轴长度）；量化细胞的形状各向异性
                'shape_anisotropy': np.nan  # 形状各向异性（1-最小特征值/最大特征值）；量化细胞的方向偏好性
            }
        else:
            morphology_data = {
                # 2D形态特征
                'area_um2': np.nan,  # 细胞面积（2D）；量化细胞的空间范围
                'convex_area_um2': np.nan,  # 细胞凸包面积（2D）；量化细胞的空间范围
                'perimeter_um': np.nan,  # 细胞周长（2D）；量化细胞的边界复杂度
                'circularity': np.nan,  # 细胞圆形度（2D）；量化细胞接近圆形的程度
                'convex_density': np.nan,  # 细胞凸包密度（细胞面积/凸包面积）；量化细胞在凸包内的填充程度
                'max_diameter_um': np.nan,  # 细胞最大直径；量化细胞的最大延伸距离
                'min_diameter_um': np.nan,  # 细胞最小直径；量化细胞的最小延伸距离
                'stretch': np.nan,  # 细胞伸展度（最大特征值-最小特征值）/最大特征值；量化细胞的形状各向异性
                'aspect_ratio': np.nan,  # 长宽比（主轴长度/最小轴长度）；量化细胞的形状各向异性
                'shape_anisotropy': np.nan  # 形状各向异性（1-最小特征值/最大特征值）；量化细胞的方向偏好性
            }

        obj_mask = cell['obj_mask']
        if obj_mask is None:
            return morphology_data

        is_3d = self.extractor.is_3d
        voxel_size = self.extractor.voxel_size

        # === 细胞基本特征 ===
        if is_3d:
            volume = np.sum(obj_mask) * np.prod(voxel_size)
            morphology_data['volume_um3'] = volume
        else:
            area = np.sum(obj_mask) * np.prod(voxel_size[1:])
            morphology_data['area_um2'] = area

        # === 细胞凸包区域 ===
        hull = morphology.convex_hull_image(obj_mask)
        if is_3d:
            convex_volume = np.sum(hull) * np.prod(voxel_size)
            morphology_data['convex_volume_um3'] = convex_volume
        else:
            convex_area = np.sum(hull) * np.prod(voxel_size[1:])
            morphology_data['convex_area_um2'] = convex_area

        # === 密度特征 ===
        if is_3d:
            if convex_volume > 0:
                density = volume / convex_volume
            else:
                density = np.nan
            morphology_data['convex_density'] = density
        else:
            if convex_area > 0:
                density = area / convex_area
            else:
                density = np.nan
            morphology_data['convex_density'] = density

        # === 表面特征 ===
        if is_3d:
            surface = 0.0
            if np.sum(obj_mask) > 0:
                try:
                    verts, faces, _, _ = measure.marching_cubes(obj_mask, spacing=voxel_size)
                    surface = measure.mesh_surface_area(verts, faces)
                except Exception:
                    surface = np.nan
            morphology_data['surface_um2'] = surface

            # 紧实度
            if surface > 0:
                compactness = (36 * np.pi * volume ** 2) / (surface ** 3)
            else:
                compactness = np.nan
            morphology_data['compactness'] = compactness
        else:
            perimeter = 0.0
            if np.sum(obj_mask) > 0:
                contours = measure.find_contours(obj_mask, 0.5)
                if contours:
                    main_contour = max(contours, key=len)
                    if len(main_contour) >= 2:
                        delta = main_contour[1:] - main_contour[:-1]
                        delta_phy = delta * voxel_size[1:]
                        perimeter = np.sum(np.linalg.norm(delta_phy, axis=1))
            morphology_data['perimeter_um'] = perimeter

            # 圆形度
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = np.nan
            morphology_data['circularity'] = circularity

        # === 形状特征 ===
        if np.sum(hull) > 0:
            coords = np.argwhere(hull)
            centroid = np.mean(coords, axis=0)
            M = coords - centroid
            S = (M.T @ M) / len(coords)

            # 特征值分解
            eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]

            # 直径计算
            mean_voxel_size = np.mean(voxel_size)
            morphology_data['max_diameter_um'] = 2 * np.sqrt(5 * eigvals[0]) * mean_voxel_size
            morphology_data['min_diameter_um'] = 2 * np.sqrt(5 * eigvals[-1]) * mean_voxel_size
            # 计算中间直径（仅3D）
            if self.extractor.is_3d and len(eigvals) > 2:
                morphology_data['med_diameter_um'] = 2 * np.sqrt(5 * eigvals[1]) * mean_voxel_size

            # 伸展度
            if eigvals[0] > 0:
                stretch = (eigvals[0] - eigvals[-1]) / eigvals[0]
            else:
                stretch = np.nan
            morphology_data['stretch'] = stretch

            # 扁平度（仅3D）
            if is_3d and len(eigvals) > 2:
                if eigvals[0] - eigvals[2] > 0:
                    oblateness = 2 * (eigvals[1] - eigvals[2]) / (eigvals[0] - eigvals[2]) - 1
                else:
                    oblateness = np.nan
                morphology_data['oblateness'] = oblateness

            # 长宽比
            if eigvals[-1] > 0:
                aspect_ratio = eigvals[0] / eigvals[-1]
            else:
                aspect_ratio = np.nan
            morphology_data['aspect_ratio'] = aspect_ratio

            # 形状各向异性
            if eigvals[0] > 0:
                shape_anisotropy = 1 - (eigvals[-1] / eigvals[0])
            else:
                shape_anisotropy = np.nan
            morphology_data['shape_anisotropy'] = shape_anisotropy

        return morphology_data

    def _calculate_cell_spatial(self, cell):
        """计算细胞空间分布特征 - 优化2D/3D处理，增加极角与角度制转换"""
        # 基础空间特征（所有维度通用）
        spatial_data = {
            'orientation_azimuth': np.nan,  # 方位角（角度）
            'orientation_vector_x': np.nan,  # 主轴方向向量x分量
            'orientation_vector_y': np.nan,  # 主轴方向向量y分量
        }

        # 3D特有特征
        if self.extractor.is_3d:
            spatial_data.update({
                'orientation_zenith': np.nan,  # 极角（角度）
                'orientation_vector_z': np.nan,  # 主轴方向向量z分量
                'zip_mean': np.nan,  # Z轴强度分布平均值（%）
                'peak_zip': np.nan,  # Z轴强度分布峰值（%）
                'peak_zip_position': np.nan,  # Z轴强度峰值位置（0-1）
            })

        obj_mask = cell['obj_mask']
        if obj_mask is None or np.sum(obj_mask) == 0:
            return spatial_data

        # 计算方向特征
        if self.extractor.is_3d:
            coords = np.argwhere(obj_mask)
            if len(coords) >= 3:
                phys_coords = coords * self.extractor.voxel_size
                pca = PCA(n_components=3)
                pca.fit(phys_coords)
                main_direction = pca.components_[0]
                dx, dy, dz = main_direction

                # 计算向量模长
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 方位角：XY平面的投影角度（弧度）
                azimuth_rad = np.arctan2(dy, dx)
                # 极角：与Z轴的夹角（弧度）
                zenith_rad = np.arccos(dz / r) if r > 0 else np.nan

                # 更新空间数据
                spatial_data.update({
                    'orientation_azimuth': np.degrees(azimuth_rad),
                    'orientation_zenith': np.degrees(zenith_rad),
                    'orientation_vector_x': dx,
                    'orientation_vector_y': dy,
                    'orientation_vector_z': dz
                })
        else:
            # 2D情况：计算方向向量
            regions = measure.regionprops(obj_mask.astype(int))
            if regions:
                orientation_angle = regions[0].orientation

                # 计算方向向量
                dx = np.cos(orientation_angle)
                dy = np.sin(orientation_angle)

                # 更新空间数据
                spatial_data.update({
                    'orientation_azimuth': np.degrees(orientation_angle),
                    'orientation_vector_x': dx,
                    'orientation_vector_y': dy
                })

        # 计算并整合ZIP特征（仅3D）
        if self.extractor.is_3d:
            zip_features = self._calculate_z_density(cell)
            spatial_data.update(zip_features)

        # 计算径向强度分布特征
        radial_features = self._calculate_radial_intensity_distribution(cell, bin_count=self.extractor.bin_count, )

        # 动态添加径向特征
        bin_count = radial_features.get('actual_bin_count', 4)
        for i in range(bin_count):
            # 径向强度分布特征
            spatial_data[f'frac_at_d_bin{i}'] = radial_features.get(f'frac_at_d_bin{i}', np.nan)  # 环内强度占比；量化蛋白在特定径向区域的分布比例
            spatial_data[f'mean_frac_bin{i}'] = radial_features.get(f'mean_frac_bin{i}', np.nan)  # 标准化平均强度；量化蛋白在特定径向区域的浓度（校正面积影响）
            spatial_data[f'radial_cv_bin{i}'] = radial_features.get(f'radial_cv_bin{i}', np.nan)  # 径向变异系数；量化蛋白在特定径向区域的方向性变异程度
            spatial_data[f'carip_bin{i}'] = radial_features.get(f'carip_bin{i}', np.nan)  # 累积径向强度分布；量化蛋白从中心到边缘的累积分布比例


        # 动态添加方向依赖特征
        sector_count = 8 if self.extractor.is_3d else 4

        # 添加扇区特征（每个环每个扇区）
        for bin_idx in range(bin_count):
            for sector_idx in range(sector_count):
                spatial_data[f'bin{bin_idx}_sector{sector_idx}_frac_at_d'] = radial_features.get(
                    f'bin{bin_idx}_sector{sector_idx}_frac_at_d', np.nan
                )  # 扇区强度占比；量化蛋白在特定径向区域和角度扇区的分布比例
                spatial_data[f'bin{bin_idx}_sector{sector_idx}_mean_frac'] = radial_features.get(
                    f'bin{bin_idx}_sector{sector_idx}_mean_frac', np.nan
                )  # 扇区标准化平均强度；量化蛋白在特定径向区域和角度扇区的浓度（校正面积影响）


        # 计算纹理特征
        texture_data = self._calculate_cell_texture(cell)
        spatial_data.update(texture_data)

        return spatial_data

    def _calculate_cell_texture(self, cell):
        """计算细胞纹理特征 - 基于Haralick纹理特征，同时提取raw_image和ridge_image的纹理特征"""
        texture_data = {}

        # 检查是否有细胞掩膜
        obj_mask = cell.get('obj_mask')
        if obj_mask is None:
            return self._get_default_texture_features()

        # 为两种图像类型分别计算纹理特征
        image_types = {
            'raw': cell.get('raw_image'),
            'ridge': cell.get('ridge_image')
        }

        # 为细胞设置更合适的距离参数 [1, 3, 5] - 更适合细胞骨架纹理
        haralick_distances = self.extractor.params.get('haralick_distance', [1, 3, 5])

        for img_type, image in image_types.items():
            if image is None:
                # 如果该类型图像不存在，设置默认NaN值
                texture_data.update(self._get_default_texture_features_for_type(img_type, haralick_distances))
                continue

            try:
                # 获取灰度级别参数，默认为256
                gray_levels = self.extractor.params.get('haralick_gray_levels', 256)

                # 准备图像数据
                pixel_data = self._prepare_image_for_texture(image, gray_levels, obj_mask)

                # 检查是否有足够的非零像素来计算纹理
                if np.sum(obj_mask) < 10:  # 至少需要10个细胞像素
                    raise ValueError(f"Insufficient cell pixels for {img_type} texture analysis")

                # 为每个距离计算纹理特征
                for distance in haralick_distances:
                    # 计算当前距离的纹理特征
                    texture_features = self._calculate_haralick_features(pixel_data, distance)

                    # 统一的特征命名（添加图像类型前缀）
                    feature_names = [
                        'angular_second_moment', 'contrast', 'correlation', 'variance',
                        'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy',
                        'entropy', 'difference_variance', 'difference_entropy', 'info_meas1', 'info_meas2'
                    ]

                    # 将特征添加到结果字典，添加距离和图像类型前缀
                    for i, feature_name in enumerate(feature_names):
                        if i < len(texture_features):
                            texture_data[f'texture_{img_type}_{feature_name}_d{distance}'] = texture_features[i]
                        else:
                            texture_data[f'texture_{img_type}_{feature_name}_d{distance}'] = np.nan

            except Exception as e:
                print(f"{img_type}图像纹理特征计算错误: {e}")
                # 如果计算失败，设置该图像类型的所有纹理特征为NaN
                texture_data.update(self._get_default_texture_features_for_type(img_type, haralick_distances))

        return texture_data

    def _prepare_image_for_texture(self, image, gray_levels, obj_mask):
        """准备图像数据用于纹理分析"""
        pixel_data = image.copy()

        # 应用细胞掩膜
        pixel_data = pixel_data * obj_mask

        # 首先将图像归一化到0-1范围
        if pixel_data.dtype == np.float32 or pixel_data.dtype == np.float64:
            # 如果已经是浮点数，确保在0-1范围内
            if pixel_data.min() < 0 or pixel_data.max() > 1:
                pixel_data = exposure.rescale_intensity(pixel_data, out_range=(0, 1))
        else:
            # 对于整数类型，先转换为浮点数再归一化
            pixel_data = pixel_data.astype(np.float64)
            pixel_data = exposure.rescale_intensity(pixel_data, out_range=(0, 1))

        # 现在安全地转换为uint8
        pixel_data = img_as_ubyte(pixel_data)

        # 如果不是256灰度级，重新缩放强度
        if gray_levels != 256:
            pixel_data = exposure.rescale_intensity(
                pixel_data,
                in_range=(0, 255),
                out_range=(0, gray_levels - 1)
            ).astype(np.uint8)

        return pixel_data

    def _get_default_texture_features(self):
        """获取默认的纹理特征字典（所有特征为NaN）"""
        default_features = {}
        image_types = ['raw', 'ridge']
        feature_names = [
            'angular_second_moment', 'contrast', 'correlation', 'variance',
            'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy',
            'entropy', 'difference_variance', 'difference_entropy', 'info_meas1', 'info_meas2'
        ]
        haralick_distances = self.extractor.params.get('haralick_distance', [1, 3, 5])

        for img_type in image_types:
            for distance in haralick_distances:
                for feature_name in feature_names:
                    default_features[f'texture_{img_type}_{feature_name}_d{distance}'] = np.nan

        return default_features

    def _get_default_texture_features_for_type(self, img_type, distances):
        """获取特定图像类型的默认纹理特征"""
        default_features = {}
        feature_names = [
            'angular_second_moment', 'contrast', 'correlation', 'variance',
            'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy',
            'entropy', 'difference_variance', 'difference_entropy', 'info_meas1', 'info_meas2'
        ]

        for distance in distances:
            for feature_name in feature_names:
                default_features[f'texture_{img_type}_{feature_name}_d{distance}'] = np.nan

        return default_features

    def _calculate_haralick_features(self, image, distance=1):
        """计算2D或3D图像的Haralick纹理特征（应用掩膜）"""
        try:
            # 处理维度问题
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image.squeeze(axis=0)

            features = mh.features.haralick(
                image,
                ignore_zeros=True,  # 忽略零值像素（背景）
                compute_14th_feature=False,
                return_mean=True,
                distance=distance
            )

            # 返回前13个特征
            return features[:13]

        except Exception as e:
            print(f"Haralick特征计算错误: {e}")
            return np.full(13, np.nan)


    def _calculate_haralick_features(self, image, distance=1):
        """计算2D或3D图像的Haralick纹理特征"""
        try:
            # 直接使用mahotas的haralick函数，它会自动处理2D和3D图像
            # 对于3D图像，它会计算13个方向的特征

            # 先判断是否是三维
            if image.ndim == 3:
                # 再判断第一个维度是否为1
                if image.shape[0] == 1:
                    # 使用squeeze去除第一个维度
                    image = image.squeeze(axis=0)  # 明确指定去除第0个维度

            features = mh.features.haralick(
                image,
                ignore_zeros=True,  # 忽略零值像素（背景）
                compute_14th_feature=False,  # 不计算第14个特征
                return_mean=True,  # 返回所有方向的平均值
                distance=distance  # 使用指定的距离
            )

            # 返回前13个特征
            return features[:13]

        except Exception as e:
            print(f"Haralick特征计算错误: {e}")
            return np.full(13, np.nan)

    def _calculate_cell_intensity(self, cell):
        """计算细胞强度特征 - 使用统一方法"""
        # 初始化所有可能的强度特征
        intensity_data = {
            'IntegratedIntensity': np.nan,
            'MeanIntensity': np.nan,
            'StdIntensity': np.nan,
            'MaxIntensity': np.nan,
            'MinIntensity': np.nan,
            'MassDisplacement_um': np.nan,
            'LowerQuartileIntensity': np.nan,
            'MedianIntensity': np.nan,
            'MADIntensity': np.nan,
            'CVIntensity': np.nan,
            'UpperQuartileIntensity': np.nan,
            'Location_CenterMassIntensity_um': np.nan,
            'Location_MaxIntensity_um': np.nan,
            'SkewnessIntensity': np.nan,
            'KurtosisIntensity': np.nan,
        }

        obj_mask = cell['obj_mask']
        if obj_mask is None or self.extractor.raw_image is None:
            return intensity_data

        # 获取细胞的所有像素点
        points = np.argwhere(obj_mask)
        if self.extractor.is_3d:
            points = [tuple(p) for p in points]  # (z, y, x)
        else:
            # 2D情况下添加z维度为0
            points = [(0, p[0], p[1]) for p in points]

        # 使用统一方法计算强度特征
        features = self.extractor._compute_unified_intensity_features(points)

        # 存储所有基础特征
        for key in intensity_data.keys():
            if key in features:
                intensity_data[key] = features[key]

        return intensity_data

    def _calculate_z_density(self, cell):
        """计算Z轴强度分布（ZIP）"""
        # 从垂直维度揭示细胞内生物分子的空间组织模式
        if not self.extractor.is_3d or cell['raw_image'] is None:
            return {}

        obj_mask = cell['obj_mask']
        raw_image = cell['raw_image']

        if obj_mask is None or np.sum(obj_mask) == 0:
            return {}

        # 1. 计算每个Z平面的强度
        z_intensities = []
        for z in range(obj_mask.shape[0]):
            # 获取当前Z平面的细胞区域
            cell_slice = obj_mask[z] & (raw_image[z] > 0)

            # 计算当前Z平面的总强度
            slice_intensity = np.sum(raw_image[z][cell_slice])
            z_intensities.append(slice_intensity)

        # 2. 计算总强度
        total_intensity = np.sum(z_intensities)

        # 3. 计算ZIP百分比
        zip_percentages = []
        for intensity in z_intensities:
            if total_intensity > 0:
                zip_percent = (intensity / total_intensity) * 100.0
            else:
                zip_percent = 0.0
            zip_percentages.append(zip_percent)

        # 4. 高度归一化（0-1范围）
        normalized_heights = np.linspace(0, 1, len(zip_percentages))

        # 5. 在100个等间距点插值
        target_points = 100
        interp_heights = np.linspace(0, 1, target_points)

        # 使用线性插值
        from scipy.interpolate import interp1d
        interp_func = interp1d(
            normalized_heights,
            zip_percentages,
            kind='linear',
            fill_value='extrapolate'
        )
        interp_zip = interp_func(interp_heights)

        # 6. 计算关键指标
        peak_value = np.max(interp_zip)
        peak_position = interp_heights[np.argmax(interp_zip)]
        mean_zip = np.mean(interp_zip)

        # 返回特征字典（只包含统计特征）
        return {
            'zip_mean': mean_zip,           # 平均ZIP值（%）；量化蛋白在Z轴的平均分布水平
            'peak_zip': peak_value,         # 最大ZIP值（%）；量化蛋白在Z轴的最大聚集强度
            'peak_zip_position': peak_position, # 峰值位置（0-1）；量化蛋白聚集最高点的相对高度（0=底部，1=顶部）
        }

    def _calculate_radial_intensity_distribution(self, cell, bin_count=4, wants_scaled=True, maximum_radius=100):
        """计算细胞径向强度分布特征 (FracAtD, MeanFrac, RadialCV, CARIP)"""
        radial_data = {'actual_bin_count': bin_count}
        obj_mask = cell['obj_mask']
        raw_image = cell['raw_image']

        if obj_mask is None or raw_image is None or np.sum(obj_mask) == 0:
            return radial_data

        is_3d = self.extractor.is_3d
        voxel_size = self.extractor.voxel_size

        # === 1. 计算距离变换 ===
        d_to_edge = ndimage.distance_transform_edt(obj_mask)

        # === 2. 计算中心点 ===
        if is_3d:
            center_coord = np.unravel_index(np.argmax(d_to_edge), obj_mask.shape)
        else:
            center_coord = np.unravel_index(np.argmax(d_to_edge), obj_mask.shape)
            center_coord = (0, center_coord[0], center_coord[1])  # 转换为3D坐标

        # === 3. 计算到中心的距离 ===
        # 创建与mask同尺寸的全零距离数组
        d_from_center = np.zeros_like(obj_mask, dtype=float)

        if is_3d:
            # 仅计算细胞区域的距离
            cell_coords = np.argwhere(obj_mask)
            if len(cell_coords) > 0:
                # 计算每个细胞点到中心的距离
                dz = cell_coords[:, 0] - center_coord[0]
                dy = cell_coords[:, 1] - center_coord[1]
                dx = cell_coords[:, 2] - center_coord[2]
                distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 将距离值填充到对应位置
                d_from_center[tuple(cell_coords.T)] = distances
        else:
            # 仅计算细胞区域的距离
            cell_coords = np.argwhere(obj_mask)
            if len(cell_coords) > 0:
                # 计算每个细胞点到中心的距离 (2D)
                dy = cell_coords[:, 0] - center_coord[1]
                dx = cell_coords[:, 1] - center_coord[2]  # 注意：center_coord在2D中是(0,y,x)
                distances = np.sqrt(dx ** 2 + dy ** 2)

                # 将距离值填充到对应位置
                d_from_center[tuple(cell_coords.T)] = distances

        # === 4. 归一化距离计算 ===
        normalized_distance = np.zeros_like(d_from_center)
        if wants_scaled:
            total_distance = d_from_center + d_to_edge
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_distance = np.where(
                    total_distance > 0,
                    d_from_center / total_distance,
                    0
                )
        else:
            normalized_distance = d_from_center / maximum_radius
            overflow_mask = (d_from_center > maximum_radius) & obj_mask
            normalized_distance[overflow_mask] = 1.0

        # === 5. 计算细胞主轴方向 ===
        if is_3d:
            # 使用PCA获取3D主轴方向
            coords = np.argwhere(obj_mask)
            if len(coords) < 3:
                return radial_data

            phys_coords = coords * voxel_size
            pca = PCA(n_components=3)
            pca.fit(phys_coords)
            main_direction = pca.components_[0]
            orientation_angle = np.arctan2(main_direction[1], main_direction[0])
        else:
            # 2D中获取方向角
            regions = measure.regionprops(obj_mask.astype(int))
            orientation_angle = regions[0].orientation if regions else 0

        # === 6. 计算CARIP特征 ===
        # 按距离排序所有像素点
        sorted_indices = np.argsort(d_from_center[obj_mask])
        sorted_intensities = raw_image[obj_mask][sorted_indices]
        total_intensity = np.sum(sorted_intensities)

        # 使用bin_count定义的分区数
        bin_edges = np.linspace(0, len(sorted_indices), bin_count + 1).astype(int)
        for i in range(bin_count):
            bin_intensity = np.sum(sorted_intensities[bin_edges[i]:bin_edges[i + 1]])
            carip = (bin_intensity / total_intensity) if total_intensity > 0 else 0
            radial_data[f'carip_bin{i}'] = carip

        # === 7. 环分配 ===
        bin_indexes = (normalized_distance * bin_count).astype(int)
        bin_indexes = np.clip(bin_indexes, 0, bin_count - 1)
        if not wants_scaled:
            bin_indexes[overflow_mask] = bin_count

        # === 8. 计算总强度和总像素数 ===
        total_intensity = np.sum(raw_image[obj_mask])
        total_pixels = np.sum(obj_mask)

        # 计算全局平均强度
        global_mean = total_intensity / total_pixels if total_pixels > 0 else 0

        # === 9. 初始化结果数组 ===
        frac_at_d = np.full(bin_count + 1, np.nan)  # +1用于溢出环
        mean_frac = np.full(bin_count + 1, np.nan)  # MeanFrac = (环内平均强度)/(全局平均强度)
        radial_cv = np.full(bin_count + 1, np.nan)

        # === 10. 计算每个环的特征 ===
        for bin_idx in range(bin_count + 1):  # 包括溢出环
            # 当前环的掩码
            bin_mask = (bin_indexes == bin_idx) & obj_mask

            bin_pixels = np.sum(bin_mask)
            if bin_pixels == 0:
                continue

            # a. 计算环内总强度
            bin_intensity = np.sum(raw_image[bin_mask])

            # b. 计算FracAtD (环内强度占比)
            if total_intensity > 0:
                frac_at_d[bin_idx] = bin_intensity / total_intensity
            else:
                frac_at_d[bin_idx] = 0

            # c. 计算MeanFrac (标准化平均强度)
            bin_mean = bin_intensity / bin_pixels if bin_pixels > 0 else 0
            if global_mean > 0:
                mean_frac[bin_idx] = bin_mean / global_mean
            else:
                mean_frac[bin_idx] = 0

            # d. 计算RadialCV (环内强度变异系数)
            # 获取环内所有点
            points = np.argwhere(bin_mask)
            if len(points) == 0:
                radial_cv[bin_idx] = np.nan
                continue

            intensities = raw_image[bin_mask]

            # 3D中使用球坐标，2D使用极坐标
            if is_3d:
                # 计算球坐标 (r, θ, φ)
                dx = points[:, 2] - center_coord[2]
                dy = points[:, 1] - center_coord[1]
                dz = points[:, 0] - center_coord[0]

                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                valid_mask = r > 0  # 只处理非中心点

                phi = np.arctan2(dy[valid_mask], dx[valid_mask])
                cos_theta = dz[valid_mask] / r[valid_mask]
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                # 初始化完整的角度数组（中心点为NaN）
                phi_full = np.full_like(dx, np.nan)
                theta_full = np.full_like(dx, np.nan)

                phi_full[valid_mask] = phi
                theta_full[valid_mask] = theta

                # 调整方位角使0度与主轴对齐
                phi_adjusted = (phi_full - orientation_angle) % (2 * np.pi)

                # 添加仰角对齐
                avg_dz = np.mean(dz[valid_mask])
                if avg_dz > 0:
                    theta_adjusted = np.pi - theta_full
                else:
                    theta_adjusted = theta_full

                # phi = np.arctan2(dy, dx)  # 方位角 [-π, π]
                # theta = np.arccos(dz / r)  # 仰角 [0, π]
                #
                # # 调整方位角使0度与主轴对齐
                # phi_adjusted = (phi - orientation_angle) % (2 * np.pi)
                #
                # # 添加仰角对齐（考虑Z轴偏移）
                # avg_dz = np.mean(dz)
                # if avg_dz > 0:
                #     theta_adjusted = np.pi - theta
                # else:
                #     theta_adjusted = theta

                # 分为8个扇区 (4方位角×2仰角)
                phi_bins = np.linspace(0, 2 * np.pi, 5)  # 4个方位分区
                theta_bins = [0, np.pi / 2, np.pi]  # 2个仰角分区

                sector_means = []
                for i in range(len(phi_bins) - 1):
                    for j in range(len(theta_bins) - 1):
                        phi_mask = (phi_adjusted >= phi_bins[i]) & (phi_adjusted < phi_bins[i + 1])
                        theta_mask = (theta_adjusted >= theta_bins[j]) & (theta_adjusted < theta_bins[j + 1])
                        sector_mask = phi_mask & theta_mask

                        if np.any(sector_mask):
                            sector_mean = np.mean(intensities[sector_mask])
                            sector_means.append(sector_mean)
                        else:
                            # 确保所有扇区都有占位符
                            sector_means.append(0)
            else:
                # 2D极坐标
                # 注意：points的形状为(n, 2)，每行是(y, x)
                dx = points[:, 1] - center_coord[2]  # x坐标
                dy = points[:, 0] - center_coord[1]  # y坐标
                phi = np.arctan2(dy, dx)  # 角度 [-π, π]

                # 调整角度使0度与主轴对齐
                phi_adjusted = (phi - orientation_angle) % (2 * np.pi)

                # 分为4个扇区
                phi_bins = np.linspace(0, 2 * np.pi, 5)

                sector_means = []
                for i in range(len(phi_bins) - 1):
                    phi_mask = (phi_adjusted >= phi_bins[i]) & (phi_adjusted < phi_bins[i + 1])
                    if np.any(phi_mask):
                        sector_mean = np.mean(intensities[phi_mask])
                        sector_means.append(sector_mean)
                    else:
                        # 确保所有扇区都有占位符
                        sector_means.append(0)

            # 计算变异系数 (确保至少有2个扇区有数据)
            valid_means = [m for m in sector_means if m > 0]
            if len(valid_means) >= 2:
                mean_val = np.mean(valid_means)
                radial_cv[bin_idx] = np.std(valid_means) / mean_val
            else:
                radial_cv[bin_idx] = 0  # 避免返回nan

            # 方向依赖特征 - 按扇区计算
            sector_count = 8 if is_3d else 4
            for sector_idx in range(sector_count):  # 修改：索引从 0 开始
                if sector_idx < len(sector_means):
                    # 获取扇区掩码
                    if is_3d:
                        i_idx = sector_idx // 2
                        j_idx = sector_idx % 2
                        phi_mask = (phi_adjusted >= phi_bins[i_idx]) & (phi_adjusted < phi_bins[i_idx + 1])
                        theta_mask = (theta_adjusted >= theta_bins[j_idx]) & (theta_adjusted < theta_bins[j_idx + 1])
                        sector_mask = phi_mask & theta_mask
                    else:
                        phi_mask = (phi_adjusted >= phi_bins[sector_idx]) & (phi_adjusted < phi_bins[sector_idx + 1])
                        sector_mask = phi_mask

                    # 计算扇区总强度
                    sector_intensity = np.sum(intensities[sector_mask]) if np.any(sector_mask) else 0

                    # 扇区强度百分比
                    if bin_intensity > 0:
                        sector_frac = sector_intensity / bin_intensity
                    else:
                        sector_frac = 0

                    radial_data[f'bin{bin_idx}_sector{sector_idx}_frac_at_d'] = sector_frac

                    # 扇区标准化平均强度
                    sector_pixels = np.sum(sector_mask)
                    if sector_pixels > 0 and bin_pixels > 0:
                        radial_data[f'bin{bin_idx}_sector{sector_idx}_mean_frac'] = sector_frac / (
                                    sector_pixels / bin_pixels)
                    else:
                        radial_data[f'bin{bin_idx}_sector{sector_idx}_mean_frac'] = 0

        # === 11. 存储FracAtD和MeanFrac结果 ===
        for bin_idx in range(bin_count):
            radial_data[f'frac_at_d_bin{bin_idx}'] = frac_at_d[bin_idx]
            radial_data[f'mean_frac_bin{bin_idx}'] = mean_frac[bin_idx]
            radial_data[f'radial_cv_bin{bin_idx}'] = radial_cv[bin_idx]

        # 溢出环处理
        if not wants_scaled:
            radial_data[f'frac_at_d_overflow'] = frac_at_d[bin_count]
            radial_data[f'mean_frac_overflow'] = mean_frac[bin_count]
            radial_data[f'radial_cv_overflow'] = radial_cv[bin_count]

        return radial_data