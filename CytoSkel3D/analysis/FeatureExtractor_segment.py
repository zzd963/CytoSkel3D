import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl

mpl.use('Agg')
from scipy import ndimage, stats

# ====================== 2. Segment Feature Calculation ======================
class SegmentFeatureCalculator:
    """Segment feature calculator"""

    def __init__(self, extractor):
        self.extractor = extractor

    def calculate_segments_features(self):
        """Calculate basic segment features: length, width, curvature, non-uniformity index (refactored version)"""
        # 1. Identify all segments (edge fragments)
        segments = self._identify_segments()

        # 2. Build mapping relationships
        self.extractor.build_mappings()

        # 3. Calculate features for each segment
        segment_features_list = []  # Store feature dictionary for each segment

        # Calculate morphological features
        morphology_data = self._calculate_segment_morphology(segments)
        # Calculate topological features
        topology_data = self._calculate_segment_topology(segments)
        # Calculate spatial distribution features
        spatial_data = self._calculate_segment_spatial(segments)
        # Calculate intensity features
        intensity_data = self._calculate_segment_intensity(segments)

        # Create a feature dictionary for each segment
        for i, segment in enumerate(segments):
            feat_dict = {
                'segment_id': i,  # Use simple numeric ID
                'branch_id': self.extractor.segment_branch_map.get(i, np.nan)
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

            segment_features_list.append(feat_dict)

        # Create non-aggregated feature table (DataFrame) - do not use index
        non_aggregated_df = pd.DataFrame(segment_features_list)

        return non_aggregated_df

    def _identify_segments(self):
        """Identify all segments (edge fragments) and add segment index"""
        segments = []
        segment_id = 0

        # Iterate through all edges and their segments
        for edge_key, properties_list in self.extractor.edge_properties.items():
            for prop_idx, props in enumerate(properties_list):
                if 'path' in props and len(props['path']) > 1:
                    # Create an enhanced edge_key containing the segment index
                    enhanced_key = (edge_key, edge_key, prop_idx)

                    segments.append({
                        'id': segment_id,
                        'edge_key': enhanced_key,  # Use enhanced key
                        'props': props,
                        'path': props['path']
                    })
                    segment_id += 1
        return segments

    def _calculate_segment_morphology(self, segments):
        """Calculate segment morphological features (physical attributes)"""
        morphology_data = {
            'length': [],  # Segment length (micrometers); quantifies physical size of segment, reflecting actual extension distance of filament
            'straight_length': [],  # Straight-line distance between endpoints (micrometers); quantifies shortest distance of segment in an ideal straight state
            'bending': [],  # Bending (dimensionless); ratio of path length to straight-line distance, quantifies bending degree of segment (value >= 1)
            'curvature': [],  # Average unsigned curvature (1/micrometer); average of curvatures at all points on path, quantifies overall bending intensity of segment
            'curvature_signed': [],  # Average signed curvature (1/micrometer); average curvature considering bending direction (positive = convex, negative = concave)
            # 'curvature_max': [],  # Maximum curvature
            'deviation': [],  # Linear deviation (micrometers); average distance from path points to the line connecting endpoints, quantifies degree of segment deviation from a straight line
            'geometric_complexity': [],  # Geometric complexity index (number of points); number of pixels contained in path, quantifies geometric complexity of segment
        }

        for segment in segments:
            path = segment['path']
            props = segment['props']

            # 1. Calculate physical length (µm) directly based on path points
            length_um = self.extractor._calculate_physical_path_length(path)
            morphology_data['length'].append(length_um)

            # 2. Calculate physical straight-line distance directly
            if len(path) >= 2:
                start_phy = self.extractor._to_physical_coordinates(path)
                end_phy = self.extractor._to_physical_coordinates(path[-1])
                straight_length_um = np.linalg.norm(np.array(end_phy) - np.array(start_phy))
            else:
                straight_length_um = 0.0
            morphology_data['straight_length'].append(straight_length_um)

            # 3. Bending (obtained from props)
            bending = props.get('bending', np.nan)
            morphology_data['bending'].append(bending)

            # 4. Curvature features
            curvatures = self._calculate_path_curvatures(path)
            unsigned_curv = np.nanmean(curvatures['unsigned']) if curvatures['unsigned'] else np.nan
            signed_curv = np.nanmean(curvatures['signed']) if curvatures['signed'] else np.nan
            morphology_data['curvature'].append(unsigned_curv)
            morphology_data['curvature_signed'].append(signed_curv)
            # morphology_data['curvature_max'].append(np.nanmax(curvatures['unsigned']))
            # Linear deviation feature
            deviation = self._calculate_linear_deviation(path)
            morphology_data['deviation'].append(deviation)

            # 5. Non-uniformity index (number of path points)
            # Referenced from "2018-Computational 3D imaging to quantify structural components and assembly of protein networks"
            geometric_complexity = len(path)
            morphology_data['geometric_complexity'].append(geometric_complexity)

        return morphology_data

    def _calculate_segment_topology(self, segments):
        """Calculate segment topological features (keeps node degree and node betweenness centrality in nodes)"""
        topology_data = {
            'edge_betweenness': []  # Edge betweenness centrality; quantifies the importance of the segment itself as a "bridge"
        }
        self._calculate_and_store_edge_betweenness()

        for segment in segments:
            edge_key = segment['edge_key']  # Keep the whole key
            props = segment['props']

            # Determine if it is a simple graph or multigraph based on key length
            if len(edge_key) == 2:
                u, v = edge_key
            else:  # Multigraph, key is (u, v, key)
                u, v, key_id = edge_key

            # 1. Edge betweenness centrality (obtained from graph attributes)
            # Try to get edge attributes (handling multigraphs)
            edge_data = {}
            # First try using the original key directly
            if edge_key in self.extractor.graph.edges:
                edge_data = self.extractor.graph.edges[edge_key]
            else:
                # Try reversing the order
                if len(edge_key) == 2:
                    reversed_key = (v, u)
                else:  # Multigraph
                    reversed_key = (v, u, key_id)

                if reversed_key in self.extractor.graph.edges:
                    edge_data = self.extractor.graph.edges[reversed_key]

            # Get edge betweenness centrality value
            edge_betweenness = edge_data.get('edge_betweenness', np.nan)
            topology_data['edge_betweenness'].append(edge_betweenness)
        return topology_data

    def _calculate_segment_spatial(self, segments):
        """Calculate segment spatial distribution features (position and orientation)"""
        if self.extractor.is_3d:
            spatial_data = {
                'azimuth_angle': [],  # Azimuth angle theta (XY plane angle); quantifies the horizontal direction of the segment
                'polar_angle': [],  # Polar angle phi (angle with Z-axis); quantifies the vertical direction of the segment (3D specific)
            }
        else:
            spatial_data = {
                'azimuth_angle': [],  # Azimuth angle theta (XY plane angle); quantifies the horizontal direction of the segment
            }

        for segment in segments:
            path = segment['path']
            props = segment['props']

            # 1. Azimuth angle (relative to the object's major axis)
            azimuth_angle = self._calculate_filament_azimuth_angle(path)
            spatial_data['azimuth_angle'].append(azimuth_angle)

            # 2. Polar angle (if 3D)
            if self.extractor.is_3d:
                polar_angle = self._calculate_filament_polar_angle(path)
                spatial_data['polar_angle'].append(polar_angle)


        return spatial_data

    def _calculate_segment_intensity(self, segments):
        """Calculate segment intensity features - use unified method"""
        # Initialization does not contain capacity
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
            'capacity': []    # Transmission capacity = average intensity / segment length, quantifies fluorescence signal intensity per unit length of filament.
        }

        for segment in segments:
            path = segment['path']
            props = segment['props']  # Use props dictionary directly

            # Use unified method to calculate intensity features
            features = self.extractor._compute_unified_intensity_features(path)

            # Store all base features
            for key in intensity_data.keys():
                if key != 'capacity':  # Handle capacity separately
                    intensity_data[key].append(features.get(key, np.nan))

            # Handle capacity feature separately
            capacity = props.get('capacity', np.nan)
            intensity_data['capacity'].append(capacity)

            # Store full feature set in segment properties
            segment['intensity_features'] = features

        return intensity_data

    def _calculate_and_store_edge_betweenness(self):
        """Calculate and store edge betweenness centrality (supports disconnected graphs)"""
        graph = self.extractor.graph
        if not graph:
            return

        try:
            if nx.is_connected(graph):
                edge_betweenness = nx.edge_betweenness_centrality(graph)
                nx.set_edge_attributes(graph, edge_betweenness, 'edge_betweenness')
            else:
                # If not a connected graph, calculate edge betweenness centrality for each connected component
                for component in nx.connected_components(graph):
                    subgraph = graph.subgraph(component)
                    edge_betweenness = nx.edge_betweenness_centrality(subgraph)
                    nx.set_edge_attributes(subgraph, edge_betweenness, 'edge_betweenness')
        except (nx.NetworkXPointlessConcept, MemoryError) as e:
            print(f"Edge betweenness centrality calculation failed: {str(e)}")
            # Set default value
            nx.set_edge_attributes(graph, 0.0, 'edge_betweenness')

    def _calculate_filament_azimuth_angle(self, path):
        """Calculate segment azimuth angle (relative to the object's major axis)"""
        if len(path) < 2:
            return np.nan

        # Calculate object orientation
        orientation_angle = self.extractor.orientation_angle

        # Calculate reference direction vector (object's major axis direction)
        ref_vector = np.array([
            np.cos(orientation_angle),
            np.sin(orientation_angle)
        ])

        # Calculate segment direction vector (from start to end point)
        start = np.array(self.extractor._to_physical_coordinates(path))
        end = np.array(self.extractor._to_physical_coordinates(path[-1]))
        vec = end - start
        length = np.linalg.norm(vec)

        if length < 1e-6:
            return np.nan

        # Take XY plane components
        if self.extractor.is_3d:
            # Note: in 3D, the order of physical coordinates is [z_phy, y_phy, x_phy]
            dy, dx = vec, vec
        else:
            if len(vec) == 3:  # (0, y, x)
                dy, dx = vec, vec
            else:  # (y, x)
                dy, dx = vec, vec

        # Calculate segment direction vector (normalized)
        filament_vector = np.array([dx, dy])
        filament_norm = np.linalg.norm(filament_vector)
        if filament_norm < 1e-6:
            return np.nan
        filament_vector = filament_vector / filament_norm

        # Calculate angle between segment and major axis
        dot_product = np.dot(ref_vector, filament_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        return np.degrees(angle)

    def _calculate_filament_polar_angle(self, path):
        """Calculate segment polar angle (3D specific, angle with Z-axis)"""
        if not self.extractor.is_3d or len(path) < 2:
            return np.nan

        # Calculate segment direction vector
        start = np.array(self.extractor._to_physical_coordinates(path))
        end = np.array(self.extractor._to_physical_coordinates(path[-1]))
        vec = end - start
        length = np.linalg.norm(vec)

        if length < 1e-6:
            return np.nan

        # In 3D, the order of physical coordinates is [z_phy, y_phy, x_phy], so the Z component is vec
        dz = vec
        # Calculate angle with Z-axis
        cos_phi = dz / length
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        return np.degrees(phi)

    def _calculate_path_curvatures(self, path):
        """Uniformly calculate path curvature features (signed, unsigned, max)"""
        curvatures = {
            'unsigned': [],  # Unsigned curvature
            'signed': []  # Signed curvature
        }

        if len(path) < 3:
            return curvatures

        # Convert to physical coordinates
        phys_path = [self.extractor._to_physical_coordinates(p) for p in path]
        phys_path = np.array(phys_path)

        # Calculate curvature at each point on the path
        for i in range(1, len(phys_path) - 1):
            p0 = phys_path[i - 1]
            p1 = phys_path[i]
            p2 = phys_path[i + 1]

            # Adjust coordinates based on dimension
            if self.extractor.is_3d:
                # 3D: Use full coordinates
                v1 = p1 - p0
                v2 = p2 - p1
            else:
                # 2D: Take only y and x components
                if len(p0) == 3:  # (z,y,x) format
                    p0 = p0[1:]
                    p1 = p1[1:]
                    p2 = p2[1:]
                v1 = p1 - p0
                v2 = p2 - p1

            # Calculate chord length (distance from p0 to p2)
            chord = np.linalg.norm(p2 - p0)
            if chord < 1e-6:
                continue

            # Calculate triangle area (using cross product)
            if self.extractor.is_3d:
                cross = np.cross(v1, v2)
                area = np.linalg.norm(cross) / 2.0
            else:
                # 2D cross product is a scalar
                area = abs(v1 * v2 - v1 * v2) / 2.0

            # Curvature = 4 * area / (chord length ^ 3)
            curvature = 4 * area / (chord ** 3) if chord > 0 else np.nan

            # Determine sign (in 2D, the sign of the cross product indicates direction)
            if self.extractor.is_3d:
                # In 3D, we use the projection on the XY plane to determine the sign
                v1_xy = v1[1:3]  # Take y,x components
                v2_xy = v2[1:3]
                cross_z = v1_xy * v2_xy - v1_xy * v2_xy
                sign = np.sign(cross_z)
            else:
                # 2D directly takes the sign of the cross product
                cross_z = v1 * v2 - v1 * v2
                sign = np.sign(cross_z)

            curvatures['unsigned'].append(abs(curvature))
            curvatures['signed'].append(curvature * sign)

        return curvatures

    def _calculate_linear_deviation(self, path):
        """Calculate linear deviation: average distance from points on the path to the line connecting the endpoints"""
        if len(path) < 3:
            return 0.0

        # Convert to physical coordinates
        phys_path = [self.extractor._to_physical_coordinates(p) for p in path]

        # Get physical coordinates of endpoints
        start = np.array(phys_path)
        end = np.array(phys_path[-1])

        # Calculate vector connecting the endpoints
        line_vector = end - start
        line_length = np.linalg.norm(line_vector)
        if line_length < 1e-6:
            return 0.0

        # Calculate distance from each point on the path to the straight line
        distances = []
        for point in phys_path:
            p = np.array(point)
            v = p - start

            # Distance formula from point to line: |(p - start) × line_vector| / |line_vector|
            if self.extractor.is_3d:
                # Use cross product for 3D
                cross = np.cross(v, line_vector)
                dist = np.linalg.norm(cross) / line_length
            else:
                # 2D case: take only y and x components
                if len(v) == 3:  # (z,y,x) format
                    v = v[1:]  # Take y,x components
                    line_vector_2d = line_vector[1:]  # Take y,x components
                else:
                    line_vector_2d = line_vector

                # 2D cross product is a scalar: v_x * line_y - v_y * line_x
                cross = v * line_vector_2d - v * line_vector_2d
                dist = abs(cross) / line_length

            distances.append(dist)

        return np.mean(distances)