import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl

mpl.use('Agg')
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from sklearn.decomposition import PCA

# ====================== 3. Branch Feature Calculation ======================
class BranchFeatureCalculator:
    """Branch Feature Calculator"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_branch_features(self):
        """Uniformly calculate branch features based on graph theory model, and return aggregated and non-aggregated feature tables"""
        # 1. Build mapping relationships
        self.extractor.build_mappings()

        # 2. Identify branch structures using dedicated function
        branches = self._identify_branches()

        # 3. Calculate features for each branch
        branch_features_list = []  # Store feature dictionary for each branch

        # Calculate morphological features
        morphology_data = self._calculate_branch_morphology(branches)
        # Calculate topological features
        topology_data = self._calculate_branch_topology(branches)
        # Calculate spatial distribution features
        spatial_data = self._calculate_branch_spatial(branches)
        # Calculate intensity features
        intensity_data = self._calculate_branch_intensity(branches)

        # Create a feature dictionary for each branch
        for i, branch in enumerate(branches):
            feat_dict = {
                'branch_id': i  # Use simple numeric ID
            }

            # Uniformly add all types of features
            feature_sources = [
                ('Morphology', morphology_data),
                ('Topology', topology_data),
                ('Spatial Distribution', spatial_data),
                ('Intensity', intensity_data)
            ]

            for category, data_dict in feature_sources:
                for key, values in data_dict.items():
                    # Safely get value using _get_value to avoid index errors
                    feat_dict[key] = self.extractor._get_value(values, i)

            branch_features_list.append(feat_dict)

        # Create non-aggregated feature table (DataFrame) - do not use index
        non_aggregated_df = pd.DataFrame(branch_features_list)

        return non_aggregated_df

    def _identify_branches(self):
        """Identify branch structures (connected components)"""
        branches = []

        # If graph does not exist, return empty list
        if not self.extractor.graph:
            return branches

        # Get connected components
        components = list(nx.connected_components(self.extractor.graph))

        for branch_id, comp in enumerate(components):
            # Create subgraph
            subgraph = self.extractor.graph.subgraph(comp)

            # Calculate branch endpoints (nodes with degree=1)
            endpoints = [n for n, d in subgraph.degree() if d == 1]

            # Calculate branch length (sum of all edge lengths)
            total_length = 0
            segments = []  # Store all segments
            branch_pixels = []  # Store all pixels of the branch (pixel coordinates)

            # Iterate through all edges
            for u, v in subgraph.edges():
                # Try two possible edge key orders
                edge_key1 = (u, v)
                edge_key2 = (v, u)

                # Check if edge properties exist
                if edge_key1 in self.extractor.edge_properties:
                    props_list = self.extractor.edge_properties[edge_key1]
                elif edge_key2 in self.extractor.edge_properties:
                    props_list = self.extractor.edge_properties[edge_key2]
                else:
                    continue

                for prop_idx, props in enumerate(props_list):
                    # Accumulate length
                    seg_length = props.get('length', 0)
                    total_length += seg_length

                    # Create segment object
                    segment = {
                        'id': f"{u}_{v}_{prop_idx}" if edge_key1 in self.extractor.edge_properties else f"{v}_{u}_{prop_idx}",
                        'edge_key': edge_key1 if edge_key1 in self.extractor.edge_properties else edge_key2,
                        'props': props
                    }
                    segments.append(segment)

                    # Collect path points (pixel coordinates)
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
                'pixels': branch_pixels  # Pixel coordinates
            })

        return branches


    def _calculate_branch_morphology(self, branches):
        """Calculate branch morphological features - basic physical properties"""
        morphology_data = {
            'length': [],  # Total branch length (physical units); quantifies branch size
            # 'segment_count': [],  # Number of segments; quantifies branch complexity
            # 'tortuosity': [],  # Tortuosity (total length / straight-line distance between endpoints); quantifies branch curvature
            'branching_factor': [],  # Branching factor (number of segments / length); quantifies branch density
            'extent': [],  # Extent (bounding box fill ratio); quantifies branch spatial utilization
            'shape_complexity': [],  # Shape complexity (entropy); quantifies irregularity of branch shape

            'aspect_ratio': [],  # Aspect ratio (major axis length / minor axis length); quantifies shape anisotropy of branch
            'axis_length_maj': [],  # Major axis length; quantifies extension of branch in primary direction
            'axis_length_min': [],  # Minor axis length; quantifies extension of branch in minor direction
            'shape_anisotropy': [],  # Shape anisotropy (1 - min eigenvalue / max eigenvalue); quantifies directional preference of branch
        }
        # 3D specific features
        if self.extractor.is_3d:
            morphology_data['axis_length_med'] = []  # Medium axis length (3D); quantifies extension of branch in secondary direction

        for branch in branches:
            # Basic morphological features
            # Accurately accumulate physical length of all segments
            branch_length_um = 0.0

            # Iterate through all segments of the branch
            for segment in branch['segments']:
                # Directly calculate physical length of each segment
                seg_path = segment['props']['path']
                branch_length_um += self.extractor._calculate_physical_path_length(seg_path)

            morphology_data['length'].append(branch_length_um)

            # # Accurately calculate branch tortuosity
            # if branch['endpoints'] and len(branch['endpoints']) >= 2:
            #     # Get physical coordinates of endpoints
            #     start_id, end_id = branch['endpoints'], branch['endpoints']
            #     start_coord = self.extractor.centroids[start_id]
            #     end_coord = self.extractor.centroids[end_id]
            #
            #     start_phy = self.extractor._to_physical_coordinates(start_coord)
            #     end_phy = self.extractor._to_physical_coordinates(end_coord)
            #
            #     # Calculate physical straight-line distance between endpoints
            #     linear_dist_um = np.linalg.norm(np.array(end_phy) - np.array(start_phy))
            #
            #     tortuosity = branch_length_um / linear_dist_um if linear_dist_um > 1e-6 else 1.0
            # else:
            #     tortuosity = np.nan
            #
            # morphology_data['tortuosity'].append(tortuosity)
            # # morphology_data['segment_count'].append(branch['segment_count'])


            # Branching factor (number of segments / length)
            if branch['total_length'] > 0:
                branching_factor = branch['segment_count'] / branch['total_length']
                morphology_data['branching_factor'].append(branching_factor)
            else:
                morphology_data['branching_factor'].append(np.nan)

            # Calculation of morphological features based on pixels
            if branch['pixels']:
                # Create binary image of branch, use uint8 type to avoid warnings
                branch_mask = np.zeros_like(self.extractor.skeleton, dtype=np.uint8)
                for pixel in branch['pixels']:
                    if self.extractor.is_3d:
                        # Ensure coordinates are integers
                        z, y, x = map(int, pixel)
                        branch_mask[z, y, x] = 1
                    else:
                        # 2D image: ensure coordinates are integers
                        if len(pixel) == 3:  # (z,y,x) format but 2D
                            _, y, x = pixel
                        else:  # (y,x) format
                            y, x = pixel
                        # Convert to integers
                        y = int(round(y))
                        x = int(round(x))
                        branch_mask[0, y, x] = 1

                # Calculate region properties
                labeled = measure.label(branch_mask)
                regions = measure.regionprops(labeled)

                if regions:
                    region = regions  # Branch has only one connected region

                    # Extent (bounding box fill ratio)
                    morphology_data['extent'].append(region.extent)

                    # Shape complexity (entropy) - Fix type conversion warning
                    if hasattr(region, 'image'):
                        # Explicitly convert to uint8 to avoid automatic conversion warnings
                        flat_image = region.image.astype(np.uint8).flatten()
                        hist = np.histogram(flat_image, bins=2)
                        hist = hist / hist.sum()
                        entropy = -np.sum(hist * np.log(hist + 1e-6))
                        morphology_data['shape_complexity'].append(entropy)

                else:
                    # Fill NaN when there are no regions
                    morphology_data['extent'].append(np.nan)
                    morphology_data['shape_complexity'].append(np.nan)
            else:
                # Fill NaN when there are no pixels
                morphology_data['extent'].append(np.nan)
                morphology_data['shape_complexity'].append(np.nan)

            # === Uniformly use eigenvalue method to calculate axis lengths ===
            # Get eigenvalues and perform stability processing
            eigenvalues = region.inertia_tensor_eigvals
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure not zero
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

            # Calculate axis lengths
            voxel_mean = np.mean(self.extractor.voxel_size)
            maj_len = 2 * np.sqrt(5 * sorted_eigenvalues) * voxel_mean
            min_len = 2 * np.sqrt(5 * sorted_eigenvalues[-1]) * voxel_mean

            morphology_data['axis_length_maj'].append(maj_len)
            morphology_data['axis_length_min'].append(min_len)

            # Uniformly calculate aspect ratio
            if min_len > 0:
                aspect_ratio = maj_len / min_len
            else:
                aspect_ratio = np.nan
            morphology_data['aspect_ratio'].append(aspect_ratio)

            # Calculate 3D specific features
            if self.extractor.is_3d:
                med_len = 2 * np.sqrt(5 * sorted_eigenvalues) * voxel_mean
                morphology_data['axis_length_med'].append(med_len)

            # Shape anisotropy = 1 - (min eigenvalue / max eigenvalue)
            if sorted_eigenvalues > 0:
                shape_anisotropy = 1 - (sorted_eigenvalues[-1] / sorted_eigenvalues)
            else:
                shape_anisotropy = np.nan
            morphology_data['shape_anisotropy'].append(shape_anisotropy)

        return morphology_data

    def _calculate_branch_topology(self, branches):
        """Calculate branch topological features - supports multigraphs"""
        topology_data = {
            'node_count': [],  # Node count; quantifies scale of branch
            'edge_count': [],  # Edge count; quantifies connection complexity of branch
            'edge_density': [],  # Edge density (actual edges / max possible edges); quantifies connection density of branch
            'global_efficiency': [],  # Global efficiency; quantifies information transmission efficiency within branch (0-1)
            'open_node_ratio': [],  # Open node ratio (ratio of degree=1 nodes); quantifies number of branch terminals
            # 'node_compactness': [],  # Node compactness (average distance / equivalent radius); quantifies clustering of nodes within branch
            'avg_branch_angle': [],  # Average angle of adjacent branches at branch points; quantifies branching morphology
        }

        for branch in branches:
            subgraph = branch['subgraph']
            n = subgraph.number_of_nodes()
            e = subgraph.number_of_edges()

            # Basic topological features
            topology_data['node_count'].append(n)
            topology_data['edge_count'].append(e)

            # Edge density = actual edges / max possible edges
            max_edges = n * (n - 1) / 2 if n > 1 else 0
            topology_data['edge_density'].append(e / max_edges if max_edges > 0 else 0)


            # Clustering coefficient and global efficiency
            try:
                if n > 0:
                    # Global efficiency - branch is a connected component, so it should be connected
                    if nx.is_connected(subgraph):
                        global_efficiency = nx.global_efficiency(subgraph)
                    else:
                        # Theoretically branch should be connected, but just in case
                        global_efficiency = np.nan
                    topology_data['global_efficiency'].append(global_efficiency)
                else:
                    topology_data['global_efficiency'].append(np.nan)
            except (nx.NetworkXError, ZeroDivisionError):
                topology_data['global_efficiency'].append(np.nan)

            # Calculate open node ratio
            degrees = [d for _, d in subgraph.degree()]
            open_node_count = sum(1 for d in degrees if d == 1)
            open_node_ratio = open_node_count / n if n > 0 else 0.0
            topology_data['open_node_ratio'].append(open_node_ratio)

            # # Calculate node compactness
            # node_compactness = np.nan
            # if n > 0:
            #     # Get branch center (physical coordinates)
            #     node_coords = []
            #     for node_id in subgraph.nodes():
            #         coord = np.array(self.extractor.centroids[node_id])
            #         node_coords.append(coord)
            #
            #     centroid_phy = np.mean(node_coords, axis=0)
            #
            #     # Calculate branch equivalent radius (using actual pixels)
            #     if self.extractor.is_3d:
            #         volume = len(branch['pixels']) * np.prod(self.extractor.voxel_size)
            #         equiv_radius = (3 * volume / (4 * np.pi)) ** (1/3)
            #     else:
            #         area = len(branch['pixels']) * np.prod(self.extractor.voxel_size)
            #         equiv_radius = np.sqrt(area / np.pi)
            #
            #     # Calculate average distance from nodes to center
            #     dists = []
            #     for coord in node_coords:
            #         dist = np.linalg.norm(coord - centroid_phy)
            #         dists.append(dist)
            #
            #     if dists and equiv_radius > 0:
            #         avg_dist = np.mean(dists)
            #         node_compactness = avg_dist / equiv_radius
            # topology_data['node_compactness'].append(node_compactness)

            # Calculate branch angles at branch points
            branch_angles = self._calculate_branch_angles(branch)
            topology_data['avg_branch_angle'].append(np.mean(branch_angles) if branch_angles else np.nan)

        return topology_data

    def _calculate_branch_spatial(self, branches):
        """Calculate branch spatial distribution features (position and orientation)"""
        spatial_data = {
            'orientation_order': []  # Orientation order parameter; quantifies consistency of segment directions within branch (0-1)
        }

        for branch in branches:
            # Initialize all features as NaN
            for key in spatial_data.keys():
                spatial_data[key].append(np.nan)

            # Skip branches with no pixels
            if not branch['pixels']:
                continue

            # Create binary image of branch
            branch_mask = np.zeros_like(self.extractor.skeleton, dtype=bool)
            for pixel in branch['pixels']:
                if self.extractor.is_3d:
                    # Ensure coordinates are integers
                    z, y, x = [int(round(coord)) for coord in pixel]
                    branch_mask[z, y, x] = True
                else:
                    if len(pixel) == 3:  # (z,y,x) format but 2D
                        _, y, x = pixel
                    else:  # (y,x) format
                        y, x = pixel
                    # Convert to integers
                    y = int(round(y))
                    x = int(round(x))
                    branch_mask[0, y, x] = True

            # Calculate region properties
            labeled = measure.label(branch_mask)
            regions = measure.regionprops(labeled)

            if not regions:
                continue

            region = regions  # Branch has only one connected region

            # === Calculate branch orientation features ===
            # Collect direction vectors of all edges
            directions = []
            for segment in branch['segments']:
                path = segment['props'].get('path', [])
                if len(path) >= 2:
                    direction = self.extractor._calculate_filament_direction(path)
                    if np.linalg.norm(direction) > 1e-6:
                        directions.append(direction)

            if directions:
                directions = np.array(directions)

                # Calculate orientation order parameter
                oop_value = self.extractor._compute_oop(directions)
                spatial_data['orientation_order'][-1] = oop_value

        return spatial_data
    def _calculate_branch_intensity(self, branches):
        """Calculate branch intensity features - using uniform method"""
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
            'capacity': [],  # Transmission capacity = average intensity / segment length, quantifies fluorescence signal intensity per unit length of filament.
        }

        for branch in branches:
            points = branch['pixels']

            # Use unified method to calculate intensity features
            features = self.extractor._compute_unified_intensity_features(points)

            # Store all 12 features
            for key in intensity_data.keys():
                if key != 'capacity':  # capacity needs separate calculation
                    intensity_data[key].append(features.get(key, np.nan))

            # Calculate branch capacity (total intensity / total length)
            total_intensity = features['IntegratedIntensity']
            total_length = branch['total_length']
            capacity = total_intensity / max(total_length, 1e-6)
            intensity_data['capacity'].append(capacity)

            # Store full feature set to branch properties
            branch['intensity_features'] = features

        return intensity_data

    def _calculate_branch_angles(self, branch):
        """Calculate branch angles at branch points"""
        branch_angles = []
        subgraph = branch['subgraph']

        for node in subgraph.nodes():
            # Only process branch points (nodes with degree >= 3)
            if subgraph.degree(node) < 3:
                continue

            # Get node voxel coordinates
            node_coord = np.array(self.extractor.centroids[node])

            # Calculate unit vector from branch point to each neighbor node
            vectors = []
            for neighbor in subgraph.neighbors(node):
                n_coord = np.array(self.extractor.centroids[neighbor])
                vec = n_coord - node_coord
                norm_val = np.linalg.norm(vec)
                if norm_val > 1e-6:
                    vectors.append(vec / norm_val)

            # Calculate angle between all adjacent branch pairs
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    dot = np.clip(np.dot(vectors[i], vectors[j]), -1, 1)
                    angle = np.arccos(dot)  # Radians
                    branch_angles.append(np.degrees(angle))  # Convert to degrees

        return branch_angles