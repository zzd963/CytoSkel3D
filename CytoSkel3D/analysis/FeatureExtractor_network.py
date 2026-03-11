import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import mahotas as mh
import skimage.util
import skimage.exposure

mpl.use('Agg')
import matplotlib.pyplot as plt
import warnings
from warnings import warn
from scipy import ndimage, stats
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from sklearn.decomposition import PCA


# ====================== 4. Network Feature Calculation ======================
class NetworkFeatureCalculator:
    """Network Feature Calculator"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_network_features(self):
        """Calculate network-level features, return aggregated feature report and non-aggregated feature table"""
        # 1. Identify network object (there is only one network)
        network = {
            'skeleton': self.extractor.skeleton,
            'obj_mask': self.extractor.obj_mask,
            'graph': self.extractor.graph
        }

        # 2. Calculate network features
        feat_dict = {}

        # Calculate morphological features
        morphology_data = self._calculate_network_morphology(network)
        # Calculate topological features
        topology_data = self._calculate_network_topology(network)
        # Calculate spatial distribution features
        spatial_data = self._calculate_network_spatial(network)
        # Calculate intensity features
        intensity_data = self._calculate_network_intensity(network)

        # Uniformly add all types of features
        feature_sources = [
            ('Morphology', morphology_data),
            ('Topology', topology_data),
            ('Spatial Distribution', spatial_data),
            ('Intensity', intensity_data)
        ]

        for category, data_dict in feature_sources:
            for key, value in data_dict.items():
                # Store values directly (not as lists)
                feat_dict[key] = value

        # Create non-aggregated feature table (DataFrame) - single row data
        non_aggregated_df = pd.DataFrame([feat_dict])

        return non_aggregated_df

    def _calculate_network_morphology(self, network):
        """Calculate network morphological features - directly return feature values (not lists)"""
        # Initialize different feature dictionaries based on dimensionality
        if self.extractor.is_3d:
            morphology_data = {
                # Network features (3D)
                'volume_um3': np.nan,  # Network volume (3D); quantifies physical size of network
                'convex_volume_um3': np.nan,  # Convex hull volume (3D); quantifies spatial extent of network
                'surface_um2': np.nan,  # Surface area (3D); quantifies surface complexity of network
                'compactness': np.nan,  # Compactness (3D); quantifies overall compactness of network

                # Shared features
                'total_length_um': np.nan,  # Total length of network
                'convex_density': np.nan,  # Network convex density (network volume / convex hull volume); quantifies filling degree of network within convex hull
                'cell_density': np.nan,  # Network cell density (network volume / cell volume); quantifies filling degree of network within cell
                'max_diameter_um': np.nan,  # Maximum diameter; quantifies maximum extension distance of network
                'med_diameter_um': np.nan,  # Median diameter (3D specific): quantifies median extension distance of network
                'min_diameter_um': np.nan,  # Minimum diameter; quantifies minimum extension distance of network
                'stretch': np.nan,  # Stretch (max eigenvalue - min eigenvalue) / max eigenvalue; quantifies shape anisotropy of network
                'oblateness': np.nan,  # Oblateness (3D); quantifies flatness of network in vertical direction
                'aspect_ratio': np.nan,  # Aspect ratio (major axis length / minor axis length); quantifies shape anisotropy of network
                'shape_anisotropy': np.nan  # Shape anisotropy (1 - min eigenvalue / max eigenvalue); quantifies directional preference of network
            }
        else:
            morphology_data = {
                # Network features (2D)
                'area_um2': np.nan,  # Network area (2D); quantifies physical size of network
                'convex_area_um2': np.nan,  # Convex hull area (2D); quantifies spatial extent of network
                'perimeter_um': np.nan,  # Perimeter (2D); quantifies boundary complexity of network
                'circularity': np.nan,  # Circularity (2D); quantifies how close network is to a perfect circle

                # Shared features
                'total_length_um': np.nan,  # Total length of network
                'convex_density': np.nan,  # Network convex density (network area / convex hull area); quantifies filling degree of network within convex hull
                'cell_density': np.nan,  # Network cell density (network area / cell area); quantifies filling degree of network within cell
                'max_diameter_um': np.nan,  # Maximum diameter; quantifies maximum extension distance of network
                'min_diameter_um': np.nan,  # Minimum diameter; quantifies minimum extension distance of network
                'stretch': np.nan,  # Stretch (max eigenvalue - min eigenvalue) / max eigenvalue; quantifies shape anisotropy of network
                # 'fractal_dimension': np.nan  # Fractal dimension
                'aspect_ratio': np.nan,  # Aspect ratio (major axis length / minor axis length); quantifies shape anisotropy of network
                'shape_anisotropy': np.nan  # Shape anisotropy (1 - min eigenvalue / max eigenvalue); quantifies directional preference of network
            }

        skeleton = network['skeleton']
        obj_mask = network['obj_mask']
        is_3d = self.extractor.is_3d
        voxel_size = self.extractor.voxel_size

        # === Basic network features ===
        if is_3d:
            volume = np.sum(skeleton) * np.prod(voxel_size)
            morphology_data['volume_um3'] = volume
        else:
            area = np.sum(skeleton) * np.prod(voxel_size)
            morphology_data['area_um2'] = area

        # === Calculate total network length ===
        # Iterate through all segments of the network
        total_length_um = 0.0
        for edge_key, properties_list in self.extractor.edge_properties.items():
            for props in properties_list:
                path = props.get('path', [])
                # total_length_um += props.get('length', [])
                total_length_um += self.extractor._calculate_physical_path_length(path)

        morphology_data['total_length_um'] = total_length_um

        # === Convex hull area ===
        if is_3d:
            hull = morphology.convex_hull_image(skeleton)
        else:
            # 2D image: ensure skeleton is a 2D array
            if skeleton.shape == 1:
                skeleton_2d = skeleton
            else:
                skeleton_2d = skeleton
            hull = morphology.convex_hull_image(skeleton_2d)
        if is_3d:
            convex_volume = np.sum(hull) * np.prod(voxel_size)
            morphology_data['convex_volume_um3'] = convex_volume
        else:
            convex_area = np.sum(hull) * np.prod(voxel_size)
            morphology_data['convex_area_um2'] = convex_area



        # === Cell region ===
        if obj_mask is not None:
            if is_3d:
                cell_volume = np.sum(obj_mask) * np.prod(voxel_size)
                morphology_data['cell_volume_um3'] = cell_volume
            else:
                cell_area = np.sum(obj_mask) * np.prod(voxel_size)
                morphology_data['cell_area_um2'] = cell_area

        # === Density features ===
        if is_3d:
            if convex_volume > 0:
                density = volume / convex_volume
            else:
                density = np.nan
            morphology_data['convex_density'] = density

            if obj_mask is not None and cell_volume > 0:
                density = volume / cell_volume
            else:
                density = np.nan
            morphology_data['cell_density'] = density
        else:
            if convex_area > 0:
                density = area / convex_area
            else:
                density = np.nan
            morphology_data['convex_density'] = density

            if obj_mask is not None and cell_area > 0:
                density = area / cell_area
            else:
                density = np.nan
            morphology_data['cell_density'] = density

        # === Surface features ===
        if is_3d:
            surface = 0.0
            if np.sum(skeleton) > 0:
                try:
                    # Use voxel size as spacing parameter
                    verts, faces, _, _ = measure.marching_cubes(skeleton, spacing=voxel_size)
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
            # 2D image: ensure skeleton is a 2D array
            if skeleton.shape == 1:
                skeleton_2d = skeleton
            else:
                skeleton_2d = skeleton

            perimeter = 0.0
            if np.sum(skeleton_2d) > 0:
                contours = measure.find_contours(skeleton_2d, 0.5)
                if contours:
                    main_contour = max(contours, key=len)
                    if len(main_contour) >= 2:
                        delta = main_contour[1:] - main_contour[:-1]
                        # Use 2D voxel size
                        if len(voxel_size) == 3:  # (z,y,x) format
                            voxel_size_2d = voxel_size[1:]
                        else:
                            voxel_size_2d = voxel_size
                        delta_phy = delta * voxel_size_2d
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

    def _calculate_network_topology(self, network):
        """Calculate network topological features - directly return feature values (not lists)"""
        topology_data = {
            'node_count': np.nan,  # Total node count; quantifies scale of network
            'edge_count': np.nan,  # Total edge count; quantifies connection complexity of network
            'edge_density': np.nan,  # Edge density (actual edges / max possible edges); quantifies connection density of branch
            'branch_ratio': np.nan,  # Branch ratio (number of branch points / total number of filaments); quantifies network complexity
            'branchpoint_density': np.nan,  # Branch point density (number of branch points / total valid pixels); quantifies density of branch structures
            'clustering_coefficient': np.nan,  # Clustering coefficient; quantifies local connection density (0-1)
            'global_efficiency': np.nan,  # Global network efficiency; measures information transmission efficiency in network (0-1)
            'open_node_ratio': np.nan,  # Open node ratio; ratio of degree 1 nodes to total nodes
            'node_compactness': np.nan,  # Node compactness; quantifies clustering degree of nodes in cell
            'R_ee': np.nan,  # Endpoint-endpoint connection ratio; quantifies direct connections of terminal structures (ratio of edges connecting two endpoints)
            'R_ej': np.nan,  # Endpoint-junction connection ratio; quantifies connection patterns of branch structures (ratio of connections between endpoints and branch points)
            'R_jj': np.nan  # Junction-junction connection ratio; quantifies core network structure (ratio of edges connecting two branch points)
        }

        graph = network['graph']
        if graph is None:
            return topology_data

        # Check if object mask is available
        obj_mask_available = network['obj_mask'] is not None and not self.extractor.full_image_mode

        # Basic graph theory properties
        num_nodes = graph.number_of_nodes()  # Use more explicit variable names
        num_edges = graph.number_of_edges()
        topology_data['node_count'] = num_nodes
        topology_data['edge_count'] = num_edges

        # Edge density calculation
        max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
        topology_data['edge_density'] = num_edges / max_edges if max_edges > 0 else 0


        # Branch ratio
        branch_points = len([n for n in graph.nodes() if graph.degree(n) >= 3])
        filaments = graph.number_of_edges()
        if filaments > 0:
            branch_ratio = branch_points / filaments
        else:
            branch_ratio = np.nan
        topology_data['branch_ratio'] = branch_ratio

        # Branch point density
        if self.extractor.obj_mask is not None:
            mask_area = np.sum(self.extractor.obj_mask) * np.prod(self.extractor.voxel_size)
        elif self.extractor.raw_image is not None:
            mask_area = np.sum(self.extractor.raw_image > np.percentile(self.extractor.raw_image, 1))
        else:
            # Use skeleton pixel count as fallback
            mask_area = np.sum(self.extractor.skeleton) * np.prod(self.extractor.voxel_size)

        if mask_area > 0:
            branchpoint_density = branch_points / mask_area
        else:
            branchpoint_density = np.nan
        topology_data['branchpoint_density'] = branchpoint_density

        # Clustering coefficient and global efficiency
        try:
            if graph.number_of_nodes() > 0:
                # Use custom method to calculate clustering coefficient for multigraphs
                valid_nodes = [n for n in graph.nodes if graph.degree(n) >= 2]
                if valid_nodes:
                    topology_data['clustering_coefficient'] = self._calculate_multigraph_clustering(valid_nodes)

                # Global efficiency - corrected calculation method
                if graph.number_of_nodes() > 1:
                    # Correct global efficiency calculation method (considers all node pairs)
                    n = graph.number_of_nodes()
                    total_pairs = n * (n - 1)

                    # Calculate efficiency between all node pairs
                    total_efficiency = 0.0
                    for i, node_i in enumerate(graph.nodes):
                        for j, node_j in enumerate(graph.nodes):
                            if i != j:
                                try:
                                    # Calculate shortest path length
                                    path_length = nx.shortest_path_length(graph, node_i, node_j)
                                    efficiency = 1 / path_length if path_length > 0 else 0
                                except nx.NetworkXNoPath:
                                    efficiency = 0  # Unreachable node pairs have an efficiency of 0
                                total_efficiency += efficiency

                    # Global efficiency = sum of all node pair efficiencies / total number of node pairs
                    global_efficiency = total_efficiency / total_pairs
                else:
                    global_efficiency = 0.0  # Single node or no nodes, global efficiency is 0
                topology_data['global_efficiency'] = global_efficiency
        except (nx.NetworkXError, ZeroDivisionError):
            pass


        # Add global node features
        # 1. Calculate open node ratio
        degrees = [d for _, d in graph.degree()]
        open_node_count = sum(1 for d in degrees if d == 1)
        total_nodes = len(degrees)
        open_node_ratio = open_node_count / total_nodes if total_nodes > 0 else 0.0
        topology_data['open_node_ratio'] = open_node_ratio

        # 2. Calculate node compactness
        if obj_mask_available:
            obj_mask = network['obj_mask']
            # Get cell center (physical coordinates)
            if obj_mask is not None:
                obj_coords = np.argwhere(obj_mask)
                if len(obj_coords) == 0:
                    centroid_phy = np.zeros(3)
                else:
                    centroid_voxel = np.mean(obj_coords, axis=0)
                    centroid_phy = centroid_voxel * self.extractor.voxel_size
            else:
                node_coords = np.array(list(self.extractor.centroids.values()))
                centroid_phy = np.mean(node_coords, axis=0) if len(node_coords) > 0 else np.zeros(3)

            # Calculate distance transform (for node-to-surface distance)
            surface_edt = None
            if obj_mask is not None:
                surface_edt = ndimage.distance_transform_edt(
                    obj_mask, sampling=self.extractor.voxel_size
                )

            # Calculate distance from node to center and from node to surface
            d_nc_list = []  # Distance from node to center
            d_ns_list = []  # Distance from node to surface

            for node_id in graph.nodes():
                coord = np.array(self.extractor.centroids[node_id])

                # Adjust node coordinates based on dimensions
                if self.extractor.is_3d:
                    coord_phy = coord
                else:
                    if len(coord) == 3:  # (z,y,x) format but 2D
                        _, y, x = coord
                        coord_phy = np.array([y, x])
                    else:  # (y,x) format
                        coord_phy = coord

                # Calculate distance from node to center
                if self.extractor.is_3d:
                    centroid_phy_adj = centroid_phy
                else:
                    centroid_phy_adj = centroid_phy if len(centroid_phy) > 2 else centroid_phy

                d_nc = np.linalg.norm(coord_phy - centroid_phy_adj)
                d_nc_list.append(d_nc)

                # Calculate distance from node to surface
                if surface_edt is not None:
                    if self.extractor.is_3d:
                        voxel_coord = np.round(coord / self.extractor.voxel_size).astype(int)
                        voxel_coord = np.clip(voxel_coord,, np.array(surface_edt.shape) - 1)
                        d_ns = surface_edt[tuple(voxel_coord)]
                    else:
                        if len(coord) == 3:  # (z,y,x) format
                            _, y, x = coord
                        else:  # (y,x) format
                            y, x = coord
                        voxel_coord = np.round(np.array([y, x]) / self.extractor.voxel_size).astype(int)
                        voxel_coord = np.clip(voxel_coord,, np.array(surface_edt.shape) - 1)
                        d_ns = surface_edt[tuple(voxel_coord)]
                    d_ns_list.append(d_ns)
                else:
                    d_ns_list.append(np.nan)

            # Calculate mean distance
            if d_nc_list:
                d_nc_mean = np.mean(d_nc_list)
            else:
                d_nc_mean = np.nan

            if d_ns_list and not np.isnan(d_ns_list).all():
                d_ns_mean = np.nanmean(d_ns_list)
            else:
                d_ns_mean = np.nan

            # Calculate compactness according to article formula
            if d_nc_mean > 0 and not np.isnan(d_ns_mean):
                compactness = (d_nc_mean - d_ns_mean) / d_nc_mean
            else:
                compactness = np.nan
            topology_data['node_compactness'] = compactness
        else:
            topology_data['node_compactness'] = np.nan
        # Count node types
        degrees = dict(graph.degree())
        endpoints = [n for n, d in degrees.items() if d == 1]  # Endpoints
        junctions = [n for n, d in degrees.items() if d > 2]  # Junctions

        # Initialize connection type counters
        ee_count = 0  # Endpoint-endpoint
        ej_count = 0  # Endpoint-junction
        jj_count = 0  # Junction-junction

        # Iterate through all edges
        for u, v in graph.edges():
            u_type = 'endpoint' if u in endpoints else ('junction' if u in junctions else 'internal')
            v_type = 'endpoint' if v in endpoints else ('junction' if v in junctions else 'internal')

            # Classify connection types
            if u_type == 'endpoint' and v_type == 'endpoint':
                ee_count += 1
            elif (u_type == 'endpoint' and v_type == 'junction') or (u_type == 'junction' and v_type == 'endpoint'):
                ej_count += 1
            elif u_type == 'junction' and v_type == 'junction':
                jj_count += 1

        # Calculate proportions
        total_edges = graph.number_of_edges()
        if total_edges > 0:
            topology_data['R_ee'] = ee_count / total_edges
            topology_data['R_ej'] = ej_count / total_edges
            topology_data['R_jj'] = jj_count / total_edges

        return topology_data

    def _calculate_network_spatial(self, network):
        """Calculate network spatial distribution features (position and orientation) - directly return feature values (not lists)"""
        spatial_data = {
            'direction_entropy': np.nan,  # Direction entropy; quantifies degree of disorder in network direction distribution
            'orientation_anisotropy': np.nan,  # Main direction dominance of network edge direction distribution, theoretical range is, higher values indicate better main direction dominance, edges are mostly aligned in similar directions
            'orientation_order': np.nan,  # Orientation order parameter (0-1); quantifies directional consistency
            'pca_comp1': np.nan,  # Variance proportion of principal component 1; describes main directional preference of network
            'pca_comp2': np.nan,  # Variance proportion of principal component 2; describes secondary directional preference of network
            'pca_comp3': np.nan,  # Variance proportion of principal component 3 (3D); describes Z-direction preference of network
            'topological_anisotropy': np.nan,  # Anisotropy of network edge spatial distribution (0-1); quantifies directional preference of network
        }

        # Calculate orientation features
        # Collect direction vectors of all edges
        directions = []
        for edge_key, properties_list in self.extractor.edge_properties.items():
            for props in properties_list:
                if 'path' in props and len(props['path']) > 1:
                    direction = self.extractor._calculate_filament_direction(props['path'])
                    if np.linalg.norm(direction) > 1e-6:
                        # Map vector to 180 degree range to eliminate directional uncertainty
                        direction_normalized = self._map_to_hemisphere(direction)
                        directions.append(direction_normalized)

        if len(directions) < 3:
            return spatial_data

        directions = np.array(directions)

        n_samples, n_features = directions.shape
        max_components = min(n_samples, n_features)
        use_components = min(3, max_components) if max_components > 0 else 0

        if use_components > 0:
            # Execute PCA
            pca = PCA(n_components=use_components).fit(directions)
            pca_var = pca.explained_variance_ratio_
            pca_vars = list(pca_var) + [np.nan] * (3 - use_components)
        else:
            pca_vars = [np.nan, np.nan, np.nan]

        # Calculate direction entropy (3D feature)
        direction_entropy = self._calc_direction_entropy(directions)

        # Calculate anisotropy index
        anisotropy = self._calculate_orientation_anisotropy(directions)

        # Calculate order parameter
        oop_value = np.nan
        if n_samples >= 3:
            oop_value = self.extractor._compute_oop(directions)

        # Store feature values
        spatial_data['direction_entropy'] = direction_entropy
        spatial_data['orientation_anisotropy'] = anisotropy
        spatial_data['orientation_order'] = oop_value
        spatial_data['pca_comp1'] = pca_vars
        spatial_data['pca_comp2'] = pca_vars
        spatial_data['pca_comp3'] = pca_vars

        # Calculate topological anisotropy
        spatial_data['topological_anisotropy'] = self._calculate_network_anisotropy()

        return spatial_data


    def _map_to_hemisphere(self, vector):
        """Map vector to hemisphere (180 degree range) to eliminate directional uncertainty"""
        vector = np.array(vector)

        if self.extractor.is_3d:
            # 3D case: ensure vector points to "upper hemisphere"
            if vector < 0:
                return -vector
            elif abs(vector) < 1e-6:
                if vector < 0:
                    return -vector
                elif abs(vector) < 1e-6 and vector < 0:
                    return -vector
        else:
            # 2D case: uniformly extract planar coordinates
            y, x = self._get_2d_components(vector)
            if x < 0:
                return -vector
            elif abs(x) < 1e-6 and y < 0:
                return -vector

        return vector

    def _get_2d_components(self, vector):
        """Uniformly extract 2D planar coordinate components"""
        vector = np.array(vector)

        if len(vector) == 3:  # (z,y,x) format
            return vector, vector  # Return (y,x)
        elif len(vector) == 2:  # (y,x) format
            return vector, vector
        else:
            # Unknown format, try to use the last two components
            return vector[-2], vector[-1]

    def _calculate_network_intensity(self, network):
        """Calculate network intensity features - directly return feature values (not lists)"""
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

        skeleton = network['skeleton']
        # Get all pixel points of the network
        points = np.argwhere(skeleton)

        # Use uniform method to calculate intensity features
        features = self.extractor._compute_unified_intensity_features(points)

        # Store all 12 features
        for key in intensity_data.keys():
            intensity_data[key] = features.get(key, np.nan)

        return intensity_data

    def _calculate_multigraph_clustering(self, nodes):
        """
        Custom method to calculate clustering coefficient for multigraphs
        Supports calculating clustering coefficient for MultiGraphs
        - Uses the analyzer's global graph object, consistent with original code
        """
        # Create a simple graph for clustering coefficient calculation
        simple_graph = nx.Graph()

        # Add all nodes
        simple_graph.add_nodes_from(self.extractor.graph.nodes)

        # Add edges (ignoring parallel edges)
        for u, v in self.extractor.graph.edges():
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v)

        # Calculate average clustering coefficient
        return nx.average_clustering(simple_graph, nodes=nodes)

    def _calculate_orientation_anisotropy(self, directions):
        """Calculate orientation anisotropy using standard principal component analysis method"""
        if len(directions) < 3:
            return np.nan

        try:
            # Calculate eigenvalues of covariance matrix
            cov_matrix = np.cov(directions.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]

            if self.extractor.is_3d:
                # 3D: Use standard anisotropy formula (λ1 - λ3) / (λ1 + λ2 + λ3)
                if len(eigenvalues) >= 3:
                    anisotropy = (eigenvalues - eigenvalues) / np.sum(eigenvalues)
                else:
                    return np.nan
            else:
                # 2D: Use 2D anisotropy formula (λ1 - λ2) / (λ1 + λ2)
                if len(eigenvalues) >= 2:
                    anisotropy = (eigenvalues - eigenvalues) / np.sum(eigenvalues[:2])
                else:
                    return np.nan

            return abs(anisotropy)  # Take absolute value to ensure non-negative

        except Exception as e:
            print(f"Anisotropy calculation error: {e}")
            return np.nan

    def _calculate_network_anisotropy(self):
        """Calculate network anisotropy index - enhanced error handling"""
        try:
            # Collect all edge vectors
            vectors = []
            for u, v in self.extractor.graph.edges():
                try:
                    node_u = np.array(self.extractor.centroids[u])
                    node_v = np.array(self.extractor.centroids[v])
                    vector = node_v - node_u

                    # Apply hemisphere mapping
                    vector_normalized = self._map_to_hemisphere(vector)
                    vectors.append(vector_normalized)
                except (KeyError, IndexError):
                    continue  # Skip invalid edges

            if len(vectors) < 2:
                return np.nan  # Requires at least 2 vectors

            vectors = np.array(vectors)

            # Calculate eigenvalues of covariance matrix
            cov_matrix = np.cov(vectors.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)

            # Calculate anisotropy index
            eigen_sum = np.sum(eigenvalues)
            if eigen_sum > 0:
                return np.max(eigenvalues) / eigen_sum
            else:
                return 0.0
        except Exception:
            return np.nan

    def _calc_direction_entropy(self, directions):
        if len(directions) < 3:
            return np.nan

        try:
            if self.extractor.is_3d:
                # 3D: Use spherical coordinates, 12 azimuth bins × 6 polar angle bins
                phi = np.arccos(np.clip(directions[:, 2], -1, 1))
                theta = np.arctan2(directions[:, 1], directions[:, 0])

                # Explicitly set number of bins
                n_theta_bins, n_phi_bins = 12, 6
                theta_bins = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
                phi_bins = np.linspace(0, np.pi, n_phi_bins + 1)

                hist, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins])
                total_bins = n_theta_bins * n_phi_bins  # 72 bins

            else:
                # 2D: Use planar angles, 12 bins
                angles = []
                for direction in directions:
                    y, x = self._get_2d_components(direction)
                    angle = np.arctan2(y, x)  # Range [-π, π]
                    angles.append(angle)

                angles = np.array(angles)
                n_bins = 12
                angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

                hist, _ = np.histogram(angles, bins=angle_bins)
                total_bins = n_bins  # 12 bins

            # Unified entropy calculation
            if hist.sum() > 0:
                hist_normalized = hist / hist.sum()
                hist_normalized = np.where(hist_normalized > 0, hist_normalized, 1e-10)
                entropy = -np.sum(hist_normalized * np.log(hist_normalized))

                max_entropy = np.log(total_bins)
                normalized_entropy = entropy / max_entropy
                return normalized_entropy
            else:
                return 0.0

        except Exception as e:
            print(f"Direction entropy calculation error: {e}")
            return np.nan