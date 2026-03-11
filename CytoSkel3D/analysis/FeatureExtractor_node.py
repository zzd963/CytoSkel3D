import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl

mpl.use('Agg')
from scipy import ndimage, stats


# ====================== 1. Node Feature Calculation ======================
class NodeFeatureCalculator:
    """Node Feature Calculator"""
    def __init__(self, extractor):
        self.extractor = extractor


    def calculate_node_features(self):
        """Calculate node-level features - refactored version, returns non-aggregated table and aggregated report"""
        # 1. Identify all nodes
        nodes = self._identify_nodes()

        # 2. Build mapping relationships
        self.extractor.build_mappings()

        # 3. Calculate features for each node
        node_features_list = []  # Store feature dictionary for each node

        # Calculate morphological features
        morphology_data = self._calculate_node_morphology(nodes)
        # Calculate topological features
        topology_data = self._calculate_node_topology(nodes)
        # Calculate spatial distribution features
        spatial_data = self._calculate_node_spatial(nodes)
        # # Calculate intensity features
        # intensity_data = self._calculate_node_intensity(nodes)

        # Create a feature dictionary for each node
        for i, node in enumerate(nodes):
            node_id = node['id']
            feat_dict = {
                'node_graph_id': node_id,  # Add node ID as first column
                'node_id': i,  # Add numeric ID for sorting
                'branch_id': self.extractor.node_branch_map.get(node_id, np.nan),
                'segment_ids': self.extractor.node_segment_map.get(node_id, [])
            }

            # Add position information (physical coordinates)
            coord = np.array(node['coord'])
            if self.extractor.is_3d:
                feat_dict.update({
                    'z_um': coord,
                    'y_um': coord,
                    'x_um': coord
                })
            else:
                if len(coord) == 3:  # (z,y,x) format but 2D
                    _, y, x = coord
                else:  # (y,x) format
                    y, x = coord
                feat_dict.update({
                    'z_um': 0,
                    'y_um': y,
                    'x_um': x
                })

            # Uniformly add all types of features
            feature_sources = [
                ('Morphology', morphology_data),
                ('Topology', topology_data),
                ('Spatial Distribution', spatial_data),
                # ('Intensity', intensity_data)
            ]

            for category, data_dict in feature_sources:
                for key, values in data_dict.items():
                    # Safely get value using get_value to avoid index errors
                    feat_dict[key] = self.extractor._get_value(values, i)

            node_features_list.append(feat_dict)

        # Create non-aggregated feature table (DataFrame) - do not use index
        non_aggregated_df = pd.DataFrame(node_features_list)

        return non_aggregated_df

    def _identify_nodes(self):
        """Identify all nodes"""
        nodes = []
        for node_id, data in self.extractor.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'coord': self.extractor.centroids[node_id],
                'data': data
            })
        return nodes

    # Morphological feature method (currently empty, can add as needed)
    def _calculate_node_morphology(self, nodes):
        """Calculate node morphological features (directly describe the physical morphological properties of the object)"""
        # Currently no specific node morphological features
        # Can add features like node size, shape, etc.
        morphology_data = {
            # Example: 'node_size': * len(nodes)
        }
        return morphology_data

    def _calculate_node_topology(self, nodes):
        """Calculate node topological features"""
        topology_data = {
            'degree': [],  # Node degree (number of connections); quantifies the connection importance of the node in the local network
            'betweenness': [],  # Node betweenness centrality; quantifies the importance of the node as a "bridge" in the network (0-1)
        }

        # Calculate node betweenness centrality (if not already calculated)
        if not hasattr(self.extractor, 'node_betweenness_calculated'):
            self._calculate_node_betweenness()
            self.extractor.node_betweenness_calculated = True

        for node in nodes:
            node_id = node['id']
            data = node['data']

            # 1. Node degree
            degree = self.extractor.graph.degree(node_id)
            topology_data['degree'].append(degree)

            # 2. Node betweenness centrality
            betweenness = data.get('node_betweenness', np.nan)
            topology_data['betweenness'].append(betweenness)

        return topology_data

    # Spatial distribution feature calculation method
    def _calculate_node_spatial(self, nodes):
        """Calculate node spatial distribution features (relative position relationships)"""
        spatial_data = {
            'to_neighbor_dist': [],  # Average distance from node to neighbor nodes (physical units); quantifies the local clustering degree of the node
            'to_surface_dist': [],  # Distance from node to cell surface (physical units); quantifies the position of the node in the cell (near surface/internal)
            'to_center_dist': [],  # Distance from node to cell center (physical units); quantifies the position of the node in the cell (near center/edge)
            'surface_to_center_ratio': []  # Ratio of surface distance to center distance; quantifies the relative position of the node (>1 indicates closer to surface)
        }

        # Check if object mask is available
        obj_mask_available = self.extractor.obj_mask is not None and not self.extractor.full_image_mode

        # === Cell center calculation ===
        centroid_phy = np.zeros(3)
        if obj_mask_available:
            obj_coords = np.argwhere(self.extractor.obj_mask)
            if len(obj_coords) > 0:
                centroid_voxel = np.mean(obj_coords, axis=0)
                centroid_phy = centroid_voxel * self.extractor.voxel_size
        else:
            # Use node coordinates to calculate center
            node_coords = np.array(list(self.extractor.centroids.values()))
            if len(node_coords) > 0:
                centroid_phy = np.mean(node_coords, axis=0)

        # === Distance transform ===
        surface_edt = None
        if obj_mask_available:
            surface_edt = ndimage.distance_transform_edt(
                self.extractor.obj_mask,
                sampling=self.extractor.voxel_size
            )

        for node in nodes:
            node_id = node['id']
            coord = np.array(node['coord'])

            # Adjust node coordinates based on dimension
            if self.extractor.is_3d:
                coord_phy = coord
            else:
                # 2D image: take only y and x components
                if len(coord) == 3:  # (z,y,x) format
                    _, y, x = coord
                else:  # (y,x) format
                    y, x = coord
                coord_phy = np.array([y, x])

            # 1. Average distance from node to neighbor nodes
            neighbors = list(self.extractor.graph.neighbors(node_id))
            to_neighbor_dist = np.nan
            if neighbors:
                dists = []
                for nid in neighbors:
                    n_coord = np.array(self.extractor.centroids[nid])
                    # Adjust neighbor coordinates based on dimension
                    if self.extractor.is_3d:
                        n_coord_phy = n_coord
                    else:
                        if len(n_coord) == 3:  # (z,y,x) format
                            _, n_y, n_x = n_coord
                            n_coord_phy = np.array([n_y, n_x])
                        else:  # (y,x) format
                            n_coord_phy = n_coord
                    dist = np.linalg.norm(coord_phy - n_coord_phy)
                    dists.append(dist)
                to_neighbor_dist = np.mean(dists)
            spatial_data['to_neighbor_dist'].append(to_neighbor_dist)

            # 2. Distance from node to cell surface
            to_surface_dist = np.nan
            if surface_edt is not None:
                if self.extractor.is_3d:
                    # Directly use voxel coordinates (pixel coordinates)
                    voxel_coord = np.round(coord).astype(int)
                    voxel_coord = np.clip(voxel_coord,, np.array(surface_edt.shape) - 1)
                    to_surface_dist = surface_edt[tuple(voxel_coord)]
                else:
                    if len(coord) == 3:  # (z,y,x) format
                        _, y, x = coord
                    else:  # (y,x) format
                        y, x = coord
                    # Directly use voxel coordinates (pixel coordinates)
                    voxel_coord = np.round(np.array([y, x])).astype(int)
                    voxel_coord = np.clip(voxel_coord,, np.array(surface_edt.shape) - 1)
                    to_surface_dist = surface_edt[tuple(voxel_coord)]
            spatial_data['to_surface_dist'].append(to_surface_dist)

            # 3. Distance from node to cell center
            # Uniformly use cell center coordinates with the same dimension as node coordinates
            if self.extractor.is_3d:
                centroid_phy_for_dist = centroid_phy
                coord_ = coord
            else:
                # 2D image: take only y and x components
                if len(centroid_phy) == 3:  # (z,y,x) format
                    _, centroid_y, centroid_x = centroid_phy
                    centroid_phy_for_dist = np.array([centroid_y, centroid_x])
                else:  # (y,x) format
                    centroid_phy_for_dist = centroid_phy
                # 2D image: take only y and x components
                if len(coord) == 3:  # (z,y,x) format
                    _, coord_y, coord_x = coord
                    coord_ = np.array([coord_y, coord_x])
                else:  # (y,x) format
                    coord_ = coord

            coord_phy = coord_ * self.extractor.voxel_size
            to_center_dist = np.linalg.norm(coord_phy - centroid_phy_for_dist)
            spatial_data['to_center_dist'].append(to_center_dist)

            # 4. Calculate ratio
            ratio = np.nan
            if to_center_dist > 1e-6 and not np.isnan(to_surface_dist):
                ratio = to_surface_dist / to_center_dist
            spatial_data['surface_to_center_ratio'].append(ratio)

        return spatial_data

    def _calculate_node_intensity(self, nodes):
        """Calculate node intensity features - use unified method"""
        intensity_data = {
            'intensity': [],  # Node intensity value (single point)
            'cv_intensity': []  # Node local intensity coefficient of variation; reflects the "purity" of the node position (whether it is in a uniform region)
        }

        for node in nodes:
            node_id = node['id']
            coord = np.array(node['coord'])  # Pixel coordinates

            # Get points in the area around the node (pixel coordinates)
            points = []

            # 3D image: 3x3x3 region
            if self.extractor.is_3d:
                z, y, x = map(int, coord)
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < self.extractor.raw_image.shape and
                                    0 <= ny < self.extractor.raw_image.shape and
                                    0 <= nx < self.extractor.raw_image.shape):
                                points.append((nz, ny, nx))
            else:
                # 2D image: 3x3 region
                if len(coord) == 3:  # (z,y,x) format
                    _, y, x = coord
                else:  # (y,x) format
                    y, x = coord
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.extractor.raw_image.shape and
                                0 <= nx < self.extractor.raw_image.shape):
                            points.append((ny, nx))

            # Use unified method to calculate intensity features (pass pixel coordinates)
            features = self.extractor._compute_unified_intensity_features(points)

            # Store node intensity value (use MeanIntensity as node intensity)
            intensity = features['MeanIntensity']
            intensity_data['intensity'].append(intensity)

            # Store node local intensity coefficient of variation
            cv_intensity = features['CVIntensity']
            intensity_data['cv_intensity'].append(cv_intensity)

            # Store full feature set to node properties
            node['intensity_features'] = features

        return intensity_data

    def _calculate_node_betweenness(self):
        """Calculate node betweenness centrality"""
        try:
            if nx.is_connected(self.extractor.graph):
                node_betweenness = nx.betweenness_centrality(self.extractor.graph)
                nx.set_node_attributes(self.extractor.graph, node_betweenness, 'node_betweenness')
            else:
                # Calculate betweenness centrality for each connected component when graph is not connected
                for component in nx.connected_components(self.extractor.graph):
                    subgraph = self.extractor.graph.subgraph(component)
                    node_betweenness = nx.betweenness_centrality(subgraph)
                    nx.set_node_attributes(self.extractor.graph, node_betweenness, 'node_betweenness')
        except (nx.NetworkXPointlessConcept, MemoryError) as e:
            print(f"Node betweenness centrality calculation failed: {str(e)}")
            nx.set_node_attributes(self.extractor.graph, 0.0, 'node_betweenness')