import os
import numpy as np
import pandas as pd
import mahotas as mh
from warnings import warn
from scipy import ndimage, stats
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from sklearn.decomposition import PCA


# ====================== 5. Cell Feature Calculation ======================
class CellFeatureCalculator:
    """Cell feature calculator (independent of network features)"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_cell_features(self):
        """Calculate cell-level features - independent of network features"""
        # Create cell object
        cell = {
            'obj_mask': self.extractor.obj_mask,
            'raw_image': self.extractor.raw_image,
            'ridge_image': self.extractor.ridge_image,
        }

        # Calculate cell features
        morphology_data = self._calculate_cell_morphology(cell)
        spatial_data = self._calculate_cell_spatial(cell)
        intensity_data = self._calculate_cell_intensity(cell)


        # Merge all features
        cell_features = {}
        cell_features.update(morphology_data)
        cell_features.update(intensity_data)
        cell_features.update(spatial_data)

        return pd.DataFrame([cell_features])

    def _calculate_cell_morphology(self, cell):
        """Calculate cell morphological features - initialize different feature dictionaries based on dimensionality"""
        # Initialize different feature dictionaries based on dimensionality
        if self.extractor.is_3d:
            morphology_data = {
                # 3D morphological features
                'volume_um3': np.nan,  # Cell volume (3D); quantifies the spatial extent of the cell
                'convex_volume_um3': np.nan,  # Cell convex hull volume (3D); quantifies the spatial extent of the cell
                'surface_um2': np.nan,  # Cell surface area (3D); quantifies the surface complexity of the cell
                'compactness': np.nan,  # Cell compactness (3D); quantifies the overall compactness of the cell
                'convex_density': np.nan,  # Cell convex density (cell volume / convex hull volume); quantifies the degree of filling within the convex hull
                'max_diameter_um': np.nan,  # Cell maximum diameter; quantifies the maximum extension distance of the cell
                'med_diameter_um': np.nan,  # Median diameter (3D specific): quantifies the median extension distance of the cell
                'min_diameter_um': np.nan,  # Cell minimum diameter; quantifies the minimum extension distance of the cell
                'stretch': np.nan,  # Cell stretch (max eigenvalue - min eigenvalue) / max eigenvalue; quantifies the shape anisotropy of the cell
                'oblateness': np.nan,  # Cell oblateness (3D); quantifies the flatness of the cell in the vertical direction
                'aspect_ratio': np.nan,  # Aspect ratio (major axis length / minor axis length); quantifies the shape anisotropy of the cell
                'shape_anisotropy': np.nan  # Shape anisotropy (1 - min eigenvalue / max eigenvalue); quantifies the directional preference of the cell
            }
        else:
            morphology_data = {
                # 2D morphological features
                'area_um2': np.nan,  # Cell area (2D); quantifies the spatial extent of the cell
                'convex_area_um2': np.nan,  # Cell convex hull area (2D); quantifies the spatial extent of the cell
                'perimeter_um': np.nan,  # Cell perimeter (2D); quantifies the boundary complexity of the cell
                'circularity': np.nan,  # Cell circularity (2D); quantifies how close the cell is to a perfect circle
                'convex_density': np.nan,  # Cell convex density (cell area / convex hull area); quantifies the degree of filling within the convex hull
                'max_diameter_um': np.nan,  # Cell maximum diameter; quantifies the maximum extension distance of the cell
                'min_diameter_um': np.nan,  # Cell minimum diameter; quantifies the minimum extension distance of the cell
                'stretch': np.nan,  # Cell stretch (max eigenvalue - min eigenvalue) / max eigenvalue; quantifies the shape anisotropy of the cell
                'aspect_ratio': np.nan,  # Aspect ratio (major axis length / minor axis length); quantifies the shape anisotropy of the cell
                'shape_anisotropy': np.nan  # Shape anisotropy (1 - min eigenvalue / max eigenvalue); quantifies the directional preference of the cell
            }

        obj_mask = cell['obj_mask']
        if obj_mask is None:
            return morphology_data

        is_3d = self.extractor.is_3d
        voxel_size = self.extractor.voxel_size

        # === Basic cell features ===
        if is_3d:
            volume = np.sum(obj_mask) * np.prod(voxel_size)
            morphology_data['volume_um3'] = volume
        else:
            area = np.sum(obj_mask) * np.prod(voxel_size[1:])
            morphology_data['area_um2'] = area

        # === Cell convex hull region ===
        hull = morphology.convex_hull_image(obj_mask)
        if is_3d:
            convex_volume = np.sum(hull) * np.prod(voxel_size)
            morphology_data['convex_volume_um3'] = convex_volume
        else:
            convex_area = np.sum(hull) * np.prod(voxel_size[1:])
            morphology_data['convex_area_um2'] = convex_area

        # === Density features ===
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

        # === Surface features ===
        if is_3d:
            surface = 0.0
            if np.sum(obj_mask) > 0:
                try:
                    verts, faces, _, _ = measure.marching_cubes(obj_mask, spacing=voxel_size)
                    surface = measure.mesh_surface_area(verts, faces)
                except Exception:
                    surface = np.nan
            morphology_data['surface_um2'] = surface

            # Compactness
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

            # Circularity
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = np.nan
            morphology_data['circularity'] = circularity

        # === Shape features ===
        if np.sum(hull) > 0:
            coords = np.argwhere(hull)
            centroid = np.mean(coords, axis=0)
            M = coords - centroid
            S = (M.T @ M) / len(coords)

            # Eigenvalue decomposition
            eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]

            # Diameter calculation
            mean_voxel_size = np.mean(voxel_size)
            morphology_data['max_diameter_um'] = 2 * np.sqrt(5 * eigvals) * mean_voxel_size
            morphology_data['min_diameter_um'] = 2 * np.sqrt(5 * eigvals[-1]) * mean_voxel_size
            # Calculate median diameter (3D only)
            if self.extractor.is_3d and len(eigvals) > 2:
                morphology_data['med_diameter_um'] = 2 * np.sqrt(5 * eigvals) * mean_voxel_size

            # Stretch
            if eigvals > 0:
                stretch = (eigvals - eigvals[-1]) / eigvals
            else:
                stretch = np.nan
            morphology_data['stretch'] = stretch

            # Oblateness (3D only)
            if is_3d and len(eigvals) > 2:
                if eigvals - eigvals > 0:
                    oblateness = 2 * (eigvals - eigvals) / (eigvals - eigvals) - 1
                else:
                    oblateness = np.nan
                morphology_data['oblateness'] = oblateness

            # Aspect ratio
            if eigvals[-1] > 0:
                aspect_ratio = eigvals / eigvals[-1]
            else:
                aspect_ratio = np.nan
            morphology_data['aspect_ratio'] = aspect_ratio

            # Shape anisotropy
            if eigvals > 0:
                shape_anisotropy = 1 - (eigvals[-1] / eigvals)
            else:
                shape_anisotropy = np.nan
            morphology_data['shape_anisotropy'] = shape_anisotropy

        return morphology_data

    def _calculate_cell_spatial(self, cell):
        """Calculate cell spatial distribution features - optimized 2D/3D processing, added polar angle and degree conversion"""
        # Basic spatial features (universal for all dimensions)
        spatial_data = {
            'orientation_azimuth': np.nan,  # Azimuth angle (degrees)
            'orientation_vector_x': np.nan,  # Main axis direction vector x-component
            'orientation_vector_y': np.nan,  # Main axis direction vector y-component
        }

        # 3D specific features
        if self.extractor.is_3d:
            spatial_data.update({
                'orientation_zenith': np.nan,  # Zenith angle (degrees)
                'orientation_vector_z': np.nan,  # Main axis direction vector z-component
                'zip_mean': np.nan,  # Z-axis intensity distribution mean (%)
                'peak_zip': np.nan,  # Z-axis intensity distribution peak (%)
                'peak_zip_position': np.nan,  # Z-axis intensity peak position (0-1)
            })

        obj_mask = cell['obj_mask']
        if obj_mask is None or np.sum(obj_mask) == 0:
            return spatial_data

        # Calculate orientation features
        if self.extractor.is_3d:
            coords = np.argwhere(obj_mask)
            if len(coords) >= 3:
                phys_coords = coords * self.extractor.voxel_size
                pca = PCA(n_components=3)
                pca.fit(phys_coords)
                main_direction = pca.components_
                dx, dy, dz = main_direction

                # Calculate vector magnitude
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # Azimuth angle: projection angle on the XY plane (radians)
                azimuth_rad = np.arctan2(dy, dx)
                # Zenith angle: angle with the Z-axis (radians)
                zenith_rad = np.arccos(dz / r) if r > 0 else np.nan

                # Update spatial data
                spatial_data.update({
                    'orientation_azimuth': np.degrees(azimuth_rad),
                    'orientation_zenith': np.degrees(zenith_rad),
                    'orientation_vector_x': dx,
                    'orientation_vector_y': dy,
                    'orientation_vector_z': dz
                })
        else:
            # 2D case: calculate orientation vector
            regions = measure.regionprops(obj_mask.astype(int))
            if regions:
                orientation_angle = regions.orientation

                # Calculate direction vector
                dx = np.cos(orientation_angle)
                dy = np.sin(orientation_angle)

                # Update spatial data
                spatial_data.update({
                    'orientation_azimuth': np.degrees(orientation_angle),
                    'orientation_vector_x': dx,
                    'orientation_vector_y': dy
                })

        # Calculate and integrate ZIP features (3D only)
        if self.extractor.is_3d:
            zip_features = self._calculate_z_density(cell)
            spatial_data.update(zip_features)

        # Calculate radial intensity distribution features
        radial_features = self._calculate_radial_intensity_distribution(cell)

        # Dynamically add radial features
        bin_count = radial_features.get('actual_bin_count', 4)
        for i in range(bin_count):
            # Radial intensity distribution features
            spatial_data[f'RIF_bin{i}'] = radial_features.get(f'RIF_bin{i}', np.nan)  # Fraction of intensity within the bin; quantifies the distribution proportion of proteins in a specific radial region
            spatial_data[f'RIE_bin{i}'] = radial_features.get(f'RIE_bin{i}', np.nan)  # Standardized mean intensity; quantifies the concentration of proteins in a specific radial region (correcting for area impact)
            spatial_data[f'RAH_bin{i}'] = radial_features.get(f'RAH_bin{i}', np.nan)  # Radial coefficient of variation; quantifies the directional variation of proteins in a specific radial region
            spatial_data[f'CRIP_bin{i}'] = radial_features.get(f'CRIP_bin{i}', np.nan)  # Cumulative radial intensity distribution; quantifies the cumulative distribution proportion of proteins from center to edge


        # Dynamically add direction-dependent features
        sector_count = 8 if self.extractor.is_3d else 4

        # Add sector features (for each bin and each sector)
        for bin_idx in range(bin_count):
            for sector_idx in range(sector_count):
                spatial_data[f'bin{bin_idx}_sector{sector_idx}_RIF'] = radial_features.get(
                    f'bin{bin_idx}_sector{sector_idx}_RIF', np.nan
                )  # Sector intensity fraction; quantifies the distribution proportion of proteins in a specific radial region and angular sector
                spatial_data[f'bin{bin_idx}_sector{sector_idx}_RIE'] = radial_features.get(
                    f'bin{bin_idx}_sector{sector_idx}_RIE', np.nan
                )  # Sector standardized mean intensity; quantifies the concentration of proteins in a specific radial region and angular sector (correcting for area impact)


        # Calculate texture features
        texture_data = self._calculate_cell_texture(cell)
        spatial_data.update(texture_data)

        return spatial_data

    def _calculate_cell_texture(self, cell):
        """Calculate cell texture features - based on Haralick texture features, extracting from both raw_image and ridge_image simultaneously"""
        texture_data = {}

        # Check if cell mask exists
        obj_mask = cell.get('obj_mask')
        if obj_mask is None:
            return self._get_default_texture_features()

        # Calculate texture features separately for two image types
        image_types = {
            'raw': cell.get('raw_image'),
            'ridge': cell.get('ridge_image')
        }

        # Set more appropriate distance parameters for cells - better suited for cytoskeleton texture
        haralick_distances = self.extractor.params.get('haralick_distance',)

        for img_type, image in image_types.items():
            if image is None:
                # If this type of image does not exist, set default NaN values
                texture_data.update(self._get_default_texture_features_for_type(img_type, haralick_distances))
                continue

            try:
                # Get gray level parameter, default is 256
                gray_levels = self.extractor.params.get('haralick_gray_levels', 256)

                # Prepare image data
                pixel_data = self._prepare_image_for_texture(image, gray_levels, obj_mask)

                # Check if there are enough non-zero pixels to calculate texture
                if np.sum(obj_mask) < 10:  # Need at least 10 cell pixels
                    raise ValueError(f"Insufficient cell pixels for {img_type} texture analysis")

                # Calculate texture features for each distance
                for distance in haralick_distances:
                    # Calculate texture features for the current distance
                    texture_features = self._calculate_haralick_features(pixel_data, distance)

                    # Unified feature naming (adding image type prefix)
                    feature_names = [
                        'angular_second_moment', 'contrast', 'correlation', 'variance',
                        'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy',
                        'entropy', 'difference_variance', 'difference_entropy', 'info_meas1', 'info_meas2'
                    ]

                    # Add features to the result dictionary, adding distance and image type prefix
                    for i, feature_name in enumerate(feature_names):
                        if i < len(texture_features):
                            texture_data[f'texture_{img_type}_{feature_name}_d{distance}'] = texture_features[i]
                        else:
                            texture_data[f'texture_{img_type}_{feature_name}_d{distance}'] = np.nan

            except Exception as e:
                print(f"Error calculating {img_type} image texture features: {e}")
                # If calculation fails, set all texture features for this image type to NaN
                texture_data.update(self._get_default_texture_features_for_type(img_type, haralick_distances))

        return texture_data

    def _prepare_image_for_texture(self, image, gray_levels, obj_mask):
        """Prepare image data for texture analysis"""
        pixel_data = image.copy()

        # Apply cell mask
        pixel_data = pixel_data * obj_mask

        # First normalize the image to 0-1 range
        if pixel_data.dtype == np.float32 or pixel_data.dtype == np.float64:
            # If already floating point, ensure it's in the 0-1 range
            if pixel_data.min() < 0 or pixel_data.max() > 1:
                pixel_data = exposure.rescale_intensity(pixel_data, out_range=(0, 1))
        else:
            # For integer types, convert to float first then normalize
            pixel_data = pixel_data.astype(np.float64)
            pixel_data = exposure.rescale_intensity(pixel_data, out_range=(0, 1))

        # Now safely convert to uint8
        pixel_data = img_as_ubyte(pixel_data)

        # Rescale intensity if not 256 gray levels
        if gray_levels != 256:
            pixel_data = exposure.rescale_intensity(
                pixel_data,
                in_range=(0, 255),
                out_range=(0, gray_levels - 1)
            ).astype(np.uint8)

        return pixel_data

    def _get_default_texture_features(self):
        """Get default texture feature dictionary (all features are NaN)"""
        default_features = {}
        image_types = ['raw', 'ridge']
        feature_names = [
            'angular_second_moment', 'contrast', 'correlation', 'variance',
            'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy',
            'entropy', 'difference_variance', 'difference_entropy', 'info_meas1', 'info_meas2'
        ]
        haralick_distances = self.extractor.params.get('haralick_distance',)

        for img_type in image_types:
            for distance in haralick_distances:
                for feature_name in feature_names:
                    default_features[f'texture_{img_type}_{feature_name}_d{distance}'] = np.nan

        return default_features

    def _get_default_texture_features_for_type(self, img_type, distances):
        """Get default texture features for a specific image type"""
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
        """Calculate Haralick texture features for 2D or 3D images (applying mask)"""
        try:
            # Handle dimension issues
            if image.ndim == 3:
                if image.shape == 1:
                    image = image.squeeze(axis=0)

            features = mh.features.haralick(
                image,
                ignore_zeros=True,  # Ignore zero-value pixels (background)
                compute_14th_feature=False,
                return_mean=True,
                distance=distance
            )

            # Return the first 13 features
            return features[:13]

        except Exception as e:
            print(f"Haralick feature calculation error: {e}")
            return np.full(13, np.nan)


    def _calculate_haralick_features(self, image, distance=1):
        """Calculate Haralick texture features for 2D or 3D images"""
        try:
            # Directly use mahotas haralick function, which automatically handles 2D and 3D images
            # For 3D images, it calculates features for 13 directions

            # First determine if it's 3D
            if image.ndim == 3:
                # Then check if the first dimension is 1
                if image.shape == 1:
                    # Use squeeze to remove the first dimension
                    image = image.squeeze(axis=0)  # Explicitly specify removing the 0th dimension

            features = mh.features.haralick(
                image,
                ignore_zeros=True,  # Ignore zero-value pixels (background)
                compute_14th_feature=False,  # Do not calculate the 14th feature
                return_mean=True,  # Return the average of all directions
                distance=distance  # Use specified distance
            )

            # Return the first 13 features
            return features[:13]

        except Exception as e:
            print(f"Haralick feature calculation error: {e}")
            return np.full(13, np.nan)

    def _calculate_cell_intensity(self, cell):
        """Calculate cell intensity features - use unified method"""
        # Initialize all possible intensity features
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

        # Get all pixel points of the cell
        points = np.argwhere(obj_mask)
        if self.extractor.is_3d:
            points = [tuple(p) for p in points]  # (z, y, x)
        else:
            # Add z-dimension as 0 for 2D cases
            points = [(0, p, p) for p in points]

        # Use unified method to calculate intensity features
        features = self.extractor._compute_unified_intensity_features(points)

        # Store all base features
        for key in intensity_data.keys():
            if key in features:
                intensity_data[key] = features[key]

        return intensity_data

    def _calculate_z_density(self, cell):
        """Calculate Z-axis intensity distribution (ZIP)"""
        # Reveal the spatial organization pattern of intracellular biomolecules from the vertical dimension
        if not self.extractor.is_3d or cell['raw_image'] is None:
            return {}

        obj_mask = cell['obj_mask']
        raw_image = cell['raw_image']

        if obj_mask is None or np.sum(obj_mask) == 0:
            return {}

        # 1. Calculate intensity for each Z plane
        z_intensities = []
        for z in range(obj_mask.shape):
            # Get cell area of current Z plane
            cell_slice = obj_mask[z] & (raw_image[z] > 0)

            # Calculate total intensity of current Z plane
            slice_intensity = np.sum(raw_image[z][cell_slice])
            z_intensities.append(slice_intensity)

        # 2. Calculate total intensity
        total_intensity = np.sum(z_intensities)

        # 3. Calculate ZIP percentage
        zip_percentages = []
        for intensity in z_intensities:
            if total_intensity > 0:
                zip_percent = (intensity / total_intensity) * 100.0
            else:
                zip_percent = 0.0
            zip_percentages.append(zip_percent)

        # 4. Normalize height (0-1 range)
        normalized_heights = np.linspace(0, 1, len(zip_percentages))

        # 5. Interpolate at 100 equally spaced points
        target_points = 100
        interp_heights = np.linspace(0, 1, target_points)

        # Use linear interpolation
        from scipy.interpolate import interp1d
        interp_func = interp1d(
            normalized_heights,
            zip_percentages,
            kind='linear',
            fill_value='extrapolate'
        )
        interp_zip = interp_func(interp_heights)

        # 6. Calculate key metrics
        peak_value = np.max(interp_zip)
        peak_position = interp_heights[np.argmax(interp_zip)]
        mean_zip = np.mean(interp_zip)

        # Return feature dictionary (only contains statistical features)
        return {
            'zip_mean': mean_zip,           # Mean ZIP value (%); quantifies average distribution level of proteins along Z-axis
            'peak_zip': peak_value,         # Maximum ZIP value (%); quantifies maximum aggregation intensity of proteins along Z-axis
            'peak_zip_position': peak_position, # Peak position (0-1); quantifies relative height of the highest protein aggregation point (0=bottom, 1=top)
        }

    def _calculate_radial_intensity_distribution(self, cell, bin_count=4, wants_scaled=True, maximum_radius=100):
        """Calculate cell radial intensity distribution features (RIF, RIE, RAH, CRIP)"""
        radial_data = {'actual_bin_count': bin_count}
        obj_mask = cell['obj_mask']
        raw_image = cell['raw_image']

        if obj_mask is None or raw_image is None or np.sum(obj_mask) == 0:
            return radial_data

        is_3d = self.extractor.is_3d
        voxel_size = self.extractor.voxel_size

        # === 1. Calculate distance transform ===
        d_to_edge = ndimage.distance_transform_edt(obj_mask)

        # === 2. Calculate center point ===
        if is_3d:
            center_coord = np.unravel_index(np.argmax(d_to_edge), obj_mask.shape)
        else:
            center_coord = np.unravel_index(np.argmax(d_to_edge), obj_mask.shape)
            center_coord = (0, center_coord, center_coord)  # Convert to 3D coordinate

        # === 3. Calculate distance to center ===
        # Create a zero-filled distance array of the same size as mask
        d_from_center = np.zeros_like(obj_mask, dtype=float)

        if is_3d:
            # Calculate distance only for cell region
            cell_coords = np.argwhere(obj_mask)
            if len(cell_coords) > 0:
                # Calculate distance from each cell point to center
                dz = cell_coords[:, 0] - center_coord
                dy = cell_coords[:, 1] - center_coord
                dx = cell_coords[:, 2] - center_coord
                distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # Fill distance values into corresponding positions
                d_from_center[tuple(cell_coords.T)] = distances
        else:
            # Calculate distance only for cell region
            cell_coords = np.argwhere(obj_mask)
            if len(cell_coords) > 0:
                # Calculate distance from each cell point to center (2D)
                dy = cell_coords[:, 0] - center_coord
                dx = cell_coords[:, 1] - center_coord  # Note: center_coord is (0,y,x) in 2D
                distances = np.sqrt(dx ** 2 + dy ** 2)

                # Fill distance values into corresponding positions
                d_from_center[tuple(cell_coords.T)] = distances

        # === 4. Calculate normalized distance ===
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

        # === 5. Calculate cell main axis direction ===
        if is_3d:
            # Use PCA to get 3D main axis direction
            coords = np.argwhere(obj_mask)
            if len(coords) < 3:
                return radial_data

            phys_coords = coords * voxel_size
            pca = PCA(n_components=3)
            pca.fit(phys_coords)
            main_direction = pca.components_
            orientation_angle = np.arctan2(main_direction, main_direction)
        else:
            # Get orientation angle in 2D
            regions = measure.regionprops(obj_mask.astype(int))
            orientation_angle = regions.orientation if regions else 0

        # === 6. Calculate CRIP features ===
        # Sort all pixels by distance
        sorted_indices = np.argsort(d_from_center[obj_mask])
        sorted_intensities = raw_image[obj_mask][sorted_indices]
        total_intensity = np.sum(sorted_intensities)

        # Partition number defined by bin_count
        bin_edges = np.linspace(0, len(sorted_indices), bin_count + 1).astype(int)
        for i in range(bin_count):
            bin_intensity = np.sum(sorted_intensities[bin_edges[i]:bin_edges[i + 1]])
            CRIP = (bin_intensity / total_intensity) if total_intensity > 0 else 0
            radial_data[f'CRIP_bin{i}'] = CRIP

        # === 7. Bin assignment ===
        bin_indexes = (normalized_distance * bin_count).astype(int)
        bin_indexes = np.clip(bin_indexes, 0, bin_count - 1)
        if not wants_scaled:
            bin_indexes[overflow_mask] = bin_count

        # === 8. Calculate total intensity and total pixels ===
        total_intensity = np.sum(raw_image[obj_mask])
        total_pixels = np.sum(obj_mask)

        # Calculate global mean intensity
        global_mean = total_intensity / total_pixels if total_pixels > 0 else 0

        # === 9. Initialize result arrays ===
        RIF = np.full(bin_count + 1, np.nan)  # +1 for overflow bin
        RIE = np.full(bin_count + 1, np.nan)  # RIE = (mean intensity in bin)/(global mean intensity)
        RAH = np.full(bin_count + 1, np.nan)

        # === 10. Calculate features for each bin ===
        for bin_idx in range(bin_count + 1):  # Including overflow bin
            # Mask for current bin
            bin_mask = (bin_indexes == bin_idx) & obj_mask

            bin_pixels = np.sum(bin_mask)
            if bin_pixels == 0:
                continue

            # a. Calculate total intensity within bin
            bin_intensity = np.sum(raw_image[bin_mask])

            # b. Calculate RIF (Radial Intensity Fraction within radial bin)
            if total_intensity > 0:
                RIF[bin_idx] = bin_intensity / total_intensity
            else:
                RIF[bin_idx] = 0

            # c. Calculate RIE (Radial Intensity Enrichment within radial bin)
            bin_mean = bin_intensity / bin_pixels if bin_pixels > 0 else 0
            if global_mean > 0:
                RIE[bin_idx] = bin_mean / global_mean
            else:
                RIE[bin_idx] = 0

            # d. Calculate RAH (Radial Angular Heterogeneity within radial bin)
            # Get all points in bin
            points = np.argwhere(bin_mask)
            if len(points) == 0:
                RAH[bin_idx] = np.nan
                continue

            intensities = raw_image[bin_mask]

            # Use spherical coordinates in 3D, polar coordinates in 2D
            if is_3d:
                # Calculate spherical coordinates (r, θ, φ)
                dx = points[:, 2] - center_coord
                dy = points[:, 1] - center_coord
                dz = points[:, 0] - center_coord

                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                valid_mask = r > 0  # Only process non-center points

                phi = np.arctan2(dy[valid_mask], dx[valid_mask])
                cos_theta = dz[valid_mask] / r[valid_mask]
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                # Initialize full angle array (center point is NaN)
                phi_full = np.full_like(dx, np.nan)
                theta_full = np.full_like(dx, np.nan)

                phi_full[valid_mask] = phi
                theta_full[valid_mask] = theta

                # Adjust azimuth angle to align 0 degrees with main axis
                phi_adjusted = (phi_full - orientation_angle) % (2 * np.pi)

                # Add elevation angle alignment
                avg_dz = np.mean(dz[valid_mask])
                if avg_dz > 0:
                    theta_adjusted = np.pi - theta_full
                else:
                    theta_adjusted = theta_full

                # phi = np.arctan2(dy, dx)  # Azimuth angle [-π, π]
                # theta = np.arccos(dz / r)  # Elevation angle [0, π]
                #
                # # Adjust azimuth angle to align 0 degrees with main axis
                # phi_adjusted = (phi - orientation_angle) % (2 * np.pi)
                #
                # # Add elevation alignment (consider Z-axis offset)
                # avg_dz = np.mean(dz)
                # if avg_dz > 0:
                #     theta_adjusted = np.pi - theta
                # else:
                #     theta_adjusted = theta

                # Divide into 8 sectors (4 azimuth × 2 elevation)
                phi_bins = np.linspace(0, 2 * np.pi, 5)  # 4 azimuth partitions
                theta_bins = [0, np.pi / 2, np.pi]  # 2 elevation partitions

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
                            # Ensure all sectors have placeholders
                            sector_means.append(0)
            else:
                # 2D polar coordinates
                # Note: points shape is (n, 2), each row is (y, x)
                dx = points[:, 1] - center_coord  # x coordinate
                dy = points[:, 0] - center_coord  # y coordinate
                phi = np.arctan2(dy, dx)  # Angle [-π, π]

                # Adjust angle to align 0 degrees with main axis
                phi_adjusted = (phi - orientation_angle) % (2 * np.pi)

                # Divide into 4 sectors
                phi_bins = np.linspace(0, 2 * np.pi, 5)

                sector_means = []
                for i in range(len(phi_bins) - 1):
                    phi_mask = (phi_adjusted >= phi_bins[i]) & (phi_adjusted < phi_bins[i + 1])
                    if np.any(phi_mask):
                        sector_mean = np.mean(intensities[phi_mask])
                        sector_means.append(sector_mean)
                    else:
                        # Ensure all sectors have placeholders
                        sector_means.append(0)

            # Calculate coefficient of variation (ensure at least 2 sectors have data)
            valid_means = [m for m in sector_means if m > 0]
            if len(valid_means) >= 2:
                mean_val = np.mean(valid_means)
                RAH[bin_idx] = np.std(valid_means) / mean_val
            else:
                RAH[bin_idx] = 0  # Avoid returning nan

            # Direction-dependent features - calculated per sector
            sector_count = 8 if is_3d else 4
            for sector_idx in range(sector_count):  # Mod: index starts from 0
                if sector_idx < len(sector_means):
                    # Get sector mask
                    if is_3d:
                        i_idx = sector_idx // 2
                        j_idx = sector_idx % 2
                        phi_mask = (phi_adjusted >= phi_bins[i_idx]) & (phi_adjusted < phi_bins[i_idx + 1])
                        theta_mask = (theta_adjusted >= theta_bins[j_idx]) & (theta_adjusted < theta_bins[j_idx + 1])
                        sector_mask = phi_mask & theta_mask
                    else:
                        phi_mask = (phi_adjusted >= phi_bins[sector_idx]) & (phi_adjusted < phi_bins[sector_idx + 1])
                        sector_mask = phi_mask

                    # Calculate total intensity of sector
                    sector_intensity = np.sum(intensities[sector_mask]) if np.any(sector_mask) else 0

                    # Sector intensity percentage
                    if bin_intensity > 0:
                        sector_frac = sector_intensity / bin_intensity
                    else:
                        sector_frac = 0

                    radial_data[f'bin{bin_idx}_sector{sector_idx}_RIF'] = sector_frac

                    # Sector standardized mean intensity
                    sector_pixels = np.sum(sector_mask)
                    if sector_pixels > 0 and bin_pixels > 0:
                        radial_data[f'bin{bin_idx}_sector{sector_idx}_RIE'] = sector_frac / (
                                    sector_pixels / bin_pixels)
                    else:
                        radial_data[f'bin{bin_idx}_sector{sector_idx}_RIE'] = 0

        # === 11. Store RIF and RIE results ===
        for bin_idx in range(bin_count):
            radial_data[f'RIF_bin{bin_idx}'] = RIF[bin_idx]
            radial_data[f'RIE_bin{bin_idx}'] = RIE[bin_idx]
            radial_data[f'RAH_bin{bin_idx}'] = RAH[bin_idx]

        # Handle overflow bin
        if not wants_scaled:
            radial_data[f'RIF_overflow'] = RIF[bin_count]
            radial_data[f'RIE_overflow'] = RIE[bin_count]
            radial_data[f'RAH_overflow'] = RAH[bin_count]

        return radial_data