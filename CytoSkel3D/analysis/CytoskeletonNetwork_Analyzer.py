# Goal: Convert biological cytoskeleton images (2D/3D) into mathematical graph theory models (NetworkX graphs) to achieve quantitative characterization of network structures

import numpy as np
import os
import threading
import networkx as nx
from scipy import ndimage
from skimage import measure
from collections import defaultdict
import tifffile
from .NetworkReconstructor import NetworkReconstructor


class CytoskeletonNetworkAnalyzer:
    """
    Optimized 2D/3D cytoskeleton network analysis module (Compute-only version)
    Responsibilities:
    1. Node identification and clustering (Super-node)
    2. Segment extraction and optimization (Topology Reconstruction)
    3. NetworkX graph construction
    """

    def __init__(self, params=None, img_info=None, object_id=None, skeleton=None, raw_image=None, object_mask=None,
                 pixel_class=None, labeled_skeleton=None, ridge_image=None,
                 voxel_size=(1, 1, 1), full_image_mode=True, **kwargs):

        self.img_info = img_info
        self.params = params
        self.raw_image = raw_image
        self.skeleton = np.asarray(skeleton).astype(bool)
        self.pixel_class = pixel_class
        self.obj_mask = object_mask
        self.labeled_skeleton = labeled_skeleton
        self.ridge_image = ridge_image

        self.object_id = object_id
        self.full_image_mode = full_image_mode
        self.voxel_size = np.array(voxel_size)
        self.if_restruct = self.params.get('if_restruct', False) if params else False

        # Dimensionality handling
        self.is_3d = self.skeleton.ndim == 3
        if not self.is_3d:
            self.skeleton = self.skeleton[np.newaxis, ...]
            if pixel_class is not None:
                self.pixel_class = pixel_class[np.newaxis, ...]
            if labeled_skeleton is not None:
                self.labeled_skeleton = labeled_skeleton[np.newaxis, ...]

        # Network structure storage
        self.graph = nx.Graph()
        self.edge_properties = {}
        self.node_properties = {}
        self.centroids = {}
        self.original_branchpoints = []
        self.super_nodes_map = {}

        # Threshold parameters
        self.angle_threshold = 30
        self.distance_threshold = 3.0
        self.reconstructor = None

        # Performance optimization
        self._distance_cache = {}
        self._lock = threading.Lock()

    def analyze_network(self, save_restruct=False):
        """
        Main analysis pipeline
        Returns: Dictionary containing intermediate segment data for use by the visualization module
        """
        endpoints, branchpoints = self._get_nodes_from_pixel_class()

        self.reconstructor = NetworkReconstructor(
            skeleton=self.skeleton,
            voxel_size=self.voxel_size,
            original_branchpoints=self.original_branchpoints,
            super_nodes_map=self.super_nodes_map,
            angle_threshold=self.angle_threshold,
            distance_threshold=self.distance_threshold
        )

        super_branchpoints = self._cluster_adjacent_branchpoints(branchpoints)
        segments, segments_bp = self._get_segments_from_labeled_skeleton()

        if self.if_restruct:
            segments_angle_guided = self.reconstructor._angle_guided_reconstruction(segments_bp)
            segments_connected = self.reconstructor._connect_discontinuous_segments(segments_angle_guided)
        else:
            segments_connected = None

        segments_use = segments_connected if self.if_restruct else segments_bp

        if self.is_3d:
            self._build_3d_network(segments_use, endpoints + branchpoints + super_branchpoints)
        else:
            self._build_2d_network(segments_use, endpoints + branchpoints + super_branchpoints)

        # File IO logic retained here (non-plotting)
        if save_restruct:
            if self.full_image_mode:
                output_dir = self.img_info.graph_dir
            else:
                output_dir = os.path.join(self.img_info.graph_dir, f"object_{self.object_id}")
            self._save_intermediate_results(segments_bp, segments_connected, output_dir)

        # Return intermediate results for external visualization
        return {
            'segments_bp': segments_bp,
            'segments_connected': segments_connected
        }

    ## Node processing module
    def _get_nodes_from_pixel_class(self):
        """Get nodes directly from pixel classification map"""
        endpoints = []
        branchpoints = []
        self.original_branchpoints = []

        coords = np.argwhere(self.pixel_class > 0)

        for coord in coords:
            if self.is_3d or coord.size == 3:
                z, y, x = coord
                pixel_value = self.pixel_class[z, y, x]
                if pixel_value == 2:  # Endpoint
                    endpoints.append((z, y, x, 'endpoint'))
                elif pixel_value == 4:  # Branch point
                    branchpoints.append((z, y, x, 'branchpoint'))
                    self.original_branchpoints.append((z, y, x))
            else:
                y, x = coord
                pixel_value = self.pixel_class[y, x]
                if pixel_value == 2:
                    endpoints.append((y, x, 'endpoint'))
                elif pixel_value == 4:
                    branchpoints.append((y, x, 'branchpoint'))
                    self.original_branchpoints.append((y, x))

        return endpoints, branchpoints

    def _cluster_adjacent_branchpoints(self, branchpoints):
        """Cluster adjacent branch points into super nodes"""
        if not branchpoints:
            return []

        shape = self.skeleton.shape
        branch_image = np.zeros(shape, dtype=bool)

        for bp in branchpoints:
            if len(bp) == 4:
                z, y, x = bp[:3]
            else:
                y, x = bp
                z = 0
            branch_image[z, y, x] = True

        structure = np.ones((3, 3, 3))
        labeled, num_clusters = ndimage.label(branch_image, structure=structure)
        super_nodes = []

        for label in range(1, num_clusters + 1):
            cluster_coords = np.argwhere(labeled == label)
            physical_coords = [self._to_physical_coordinates(tuple(coord)) for coord in cluster_coords]
            avg_physical = np.mean(physical_coords, axis=0)
            center_index = tuple(np.round(avg_physical / self.voxel_size).astype(int))

            center_is_branch = False
            for coord in cluster_coords:
                if np.array_equal(coord, center_index):
                    center_is_branch = True
                    break

            if not center_is_branch:
                min_dist = float('inf')
                closest_coord = None
                for coord in cluster_coords:
                    dist = np.linalg.norm(np.array(coord) - np.array(center_index))
                    if dist < min_dist:
                        min_dist = dist
                        closest_coord = coord
                if closest_coord is not None:
                    center_index = tuple(closest_coord)

            if self.is_3d:
                super_coord = center_index
                for coord in cluster_coords:
                    self.super_nodes_map[tuple(coord)] = super_coord
                super_nodes.append((super_coord[0], super_coord[1], super_coord[2], 'super_branchpoint'))
            else:
                super_coord = (center_index[1], center_index[2])
                for coord in cluster_coords:
                    self.super_nodes_map[(coord[1], coord[2])] = super_coord
                super_nodes.append((super_coord[0], super_coord[1], 'super_branchpoint'))

        return super_nodes

    ## Segment processing module
    def _get_segments_from_labeled_skeleton(self):
        """Get segments directly from labeled skeleton"""
        temp_skeleton = self.skeleton.copy()
        branch_mask = self.pixel_class == 4

        if self.is_3d:
            temp_skeleton[branch_mask] = False
            connectivity = 3
        else:
            temp_skeleton[branch_mask] = False
            connectivity = 2

        labeled = measure.label(temp_skeleton, connectivity=connectivity)
        regions = measure.regionprops(labeled)

        segments = []
        segments_bp = []

        for region in regions:
            path = region.coords
            endpoints = []

            # Simplified endpoint detection logic
            for point in path:
                if self.is_3d:
                    z, y, x = point
                    z_start, z_end = max(z - 1, 0), min(z + 2, self.skeleton.shape[0])
                    y_start, y_end = max(y - 1, 0), min(y + 2, self.skeleton.shape[1])
                    x_start, x_end = max(x - 1, 0), min(x + 2, self.skeleton.shape[2])
                    neighborhood = temp_skeleton[z_start:z_end, y_start:y_end, x_start:x_end]
                else:
                    z, y, x = point
                    y_start, y_end = max(y - 1, 0), min(y + 2, self.skeleton.shape[1])
                    x_start, x_end = max(x - 1, 0), min(x + 2, self.skeleton.shape[2])
                    neighborhood = temp_skeleton[0, y_start:y_end, x_start:x_end]

                skeleton_neighbors = np.sum(neighborhood) - 1
                if skeleton_neighbors == 1:
                    endpoints.append((z, y, x))

            if len(endpoints) < 2:
                continue

            # Find the original branch point positions connected by the segment
            connected_original_bps = []
            connected_super_nodes = []

            for point in path:
                if self.is_3d:
                    z, y, x = point
                    for dz in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                nz, ny, nx = z + dz, y + dy, x + dx
                                if (0 <= nz < self.skeleton.shape[0] and
                                        0 <= ny < self.skeleton.shape[1] and
                                        0 <= nx < self.skeleton.shape[2] and
                                        self.pixel_class[nz, ny, nx] == 4):
                                    orig_coord = (nz, ny, nx)
                                    super_coord = self.super_nodes_map.get(orig_coord, orig_coord)
                                    connected_original_bps.append(orig_coord)
                                    connected_super_nodes.append(super_coord)
                else:
                    z, y, x = point
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.skeleton.shape[1] and
                                    0 <= nx < self.skeleton.shape[2] and
                                    self.pixel_class[0, ny, nx] == 4):
                                orig_coord = (0, ny, nx)
                                orig_2d = (ny, nx)
                                super_coord = self.super_nodes_map.get(orig_2d, orig_2d)
                                if len(super_coord) == 2:
                                    super_coord_3d = (0, super_coord[0], super_coord[1])
                                else:
                                    super_coord_3d = super_coord
                                connected_original_bps.append(orig_coord)
                                connected_super_nodes.append(super_coord_3d)

            seg_data = {
                'start_point': endpoints[0],
                'end_point': endpoints[1],
                'path': path,
                'label': region.label,
                'connected_original_bps': list(set(connected_original_bps)),
                'connected_super_nodes': list(set(connected_super_nodes))
            }
            seg_data['path'] = self._order_path_points_dijkstra(seg_data['path'], seg_data['start_point'],
                                                                seg_data['end_point'])
            segments.append(seg_data)

            # Create segments connecting super nodes
            seg_bp = seg_data.copy()
            del seg_bp['connected_original_bps']

            for orig_bp, super_node in zip(connected_original_bps, connected_super_nodes):
                start_dist = self._calculate_distance(orig_bp, seg_bp['start_point'])
                end_dist = self._calculate_distance(orig_bp, seg_bp['end_point'])

                if start_dist > np.sqrt(3) + 1e-5 and end_dist > np.sqrt(3) + 1e-5:
                    continue

                if start_dist < end_dist:
                    bp_to_super_path = self.reconstructor._find_shortest_path_on_skeleton(super_node, orig_bp)
                    new_path = np.vstack([bp_to_super_path, seg_bp['path']])
                    seg_bp['path'] = new_path
                    seg_bp['start_point'] = tuple(super_node)
                else:
                    bp_to_super_path = self.reconstructor._find_shortest_path_on_skeleton(orig_bp, super_node)
                    new_path = np.vstack([seg_bp['path'], bp_to_super_path])
                    seg_bp['path'] = new_path
                    seg_bp['end_point'] = tuple(super_node)

            seg_bp['path'] = self._order_path_points_dijkstra(seg_bp['path'], seg_bp['start_point'],
                                                              seg_bp['end_point'])
            segments_bp.append(seg_bp)

        return segments, segments_bp

    def _order_path_points_dijkstra(self, points, start_point, end_point):
        """Optimize path ordering using Dijkstra's algorithm"""
        if len(points) < 3:
            return points

        try:
            all_points = points.tolist() if isinstance(points, np.ndarray) else list(points)
            start_in = any(np.array_equal(start_point, p) for p in all_points)
            end_in = any(np.array_equal(end_point, p) for p in all_points)

            if not start_in: all_points.append(start_point)
            if not end_in: all_points.append(end_point)

            n = len(all_points)
            graph = defaultdict(list)
            for i in range(n):
                for j in range(i + 1, n):
                    if self._calculate_distance(all_points[i], all_points[j]) <= np.sqrt(3) + 1e-5:
                        graph[i].append(j)
                        graph[j].append(i)

            start_idx = next(i for i, p in enumerate(all_points) if np.array_equal(p, start_point))
            end_idx = next(i for i, p in enumerate(all_points) if np.array_equal(p, end_point))

            dist = {i: float('inf') for i in range(n)}
            prev = {i: -1 for i in range(n)}
            dist[start_idx] = 0
            queue = [(0, start_idx)]
            import heapq

            while queue:
                d, u = heapq.heappop(queue)
                if d > dist[u]: continue
                if u == end_idx: break

                for v in graph[u]:
                    alt = dist[u] + 1
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        heapq.heappush(queue, (alt, v))

            path_indices = []
            u = end_idx
            while u != -1:
                path_indices.append(u)
                u = prev[u]

            ordered_points = [all_points[i] for i in reversed(path_indices)]
            return ordered_points
        except Exception as e:
            print(f"Path ordering failed: {e}")
            return points

    ## Network construction module
    def _build_3d_network(self, segments, nodes):
        """Build 3D network"""
        node_id_map = {}
        next_node_id = 0
        self.graph = nx.MultiGraph()
        self.edge_properties = defaultdict(list)

        node_positions = set()
        for seg in segments:
            node_positions.add(tuple(seg['start_point']))
            node_positions.add(tuple(seg['end_point']))

        for point in node_positions:
            point_tuple = tuple(point)
            if point_tuple not in node_id_map:
                node_id = f"node_{next_node_id}"
                node_id_map[point_tuple] = node_id
                phys_coord = self._to_physical_coordinates(point)
                node_type = 'branchpoint' if point in self.original_branchpoints else 'endpoint'
                self.graph.add_node(node_id, z=phys_coord[0], y=phys_coord[1], x=phys_coord[2],
                                    pos=phys_coord, type=node_type)
                self.centroids[node_id] = phys_coord
                next_node_id += 1

        for seg in segments:
            self._add_single_edge(seg, node_id_map)

    def _build_2d_network(self, segments, nodes):
        """Build 2D network"""
        node_id_map = {}
        next_node_id = 0
        self.graph = nx.MultiGraph()
        self.edge_properties = defaultdict(list)

        for seg in segments:
            for point in [seg['start_point'], seg['end_point']]:
                point_tuple = tuple(point)
                if point_tuple not in node_id_map:
                    node_id = f"node_{next_node_id}"
                    node_id_map[point_tuple] = node_id
                    phys_coord = self._to_physical_coordinates(point)
                    node_type = 'branchpoint' if point in self.original_branchpoints else 'endpoint'
                    self.graph.add_node(node_id, z=phys_coord[0], y=phys_coord[1], x=phys_coord[2],
                                        pos=phys_coord, type=node_type)
                    self.centroids[node_id] = phys_coord
                    next_node_id += 1

        for seg in segments:
            self._add_single_edge(seg, node_id_map)

    def _add_single_edge(self, seg, node_id_map):
        start = tuple(seg['start_point'])
        end = tuple(seg['end_point'])
        if start not in node_id_map or end not in node_id_map: return

        start_id, end_id = node_id_map[start], node_id_map[end]
        ordered_path = seg.get('path', [])
        if len(ordered_path) == 0: return

        physical_path = self._to_physical_coordinates(ordered_path)
        path_length = self._calculate_physical_path_length(physical_path)
        start_phys, end_phys = self._to_physical_coordinates(start), self._to_physical_coordinates(end)
        straight_length = self._calculate_physical_distance(start_phys, end_phys)
        bending = path_length / straight_length if straight_length > 0 else 1.0
        capacity = self._calculate_edge_capacity(seg, path_length)

        self.graph.add_edge(start_id, end_id, length=path_length, straight_length=straight_length,
                            capacity=capacity, bending=bending)
        self.edge_properties[(start_id, end_id)].append({
            'path': physical_path, 'length': path_length,
            'straight_length': straight_length, 'bending': bending, 'capacity': capacity
        })

    def _calculate_distance(self, p1, p2):
        p1_arr, p2_arr = np.array(p1), np.array(p2)
        return np.linalg.norm(p1_arr - p2_arr)

    def _calculate_physical_path_length(self, physical_path):
        if len(physical_path) < 2: return 0.0
        total_length = 0.0
        for i in range(1, len(physical_path)):
            total_length += self._calculate_physical_distance(physical_path[i - 1], physical_path[i])
        return total_length

    def _calculate_physical_distance(self, pos1, pos2):
        pos1_phy, pos2_phy = self._to_physical_coordinates(pos1), self._to_physical_coordinates(pos2)
        dx, dy, dz = pos2_phy[2] - pos1_phy[2], pos2_phy[1] - pos1_phy[1], pos2_phy[0] - pos1_phy[0]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def _calculate_edge_capacity(self, data, path_length):
        if self.raw_image is None: return 0.0
        path = data.get('path', [])
        intensities = []
        for point in path:
            try:
                if self.is_3d:
                    z, y, x = map(int, point[:3])
                    intensities.append(self.raw_image[z, y, x])
                else:
                    z, y, x = point if len(point) == 3 else (0, point[0], point[1])
                    intensities.append(self.raw_image[y, x])
            except (IndexError, ValueError):
                continue
        avg_intensity = np.mean(intensities) if intensities else 0
        return avg_intensity / max(path_length, 1e-6)

    def _to_physical_coordinates(self, input_data):
        if isinstance(input_data, (tuple, list, np.ndarray)) and not isinstance(input_data[0],
                                                                                (tuple, list, np.ndarray)):
            return self._convert_single_point(input_data)
        return [self._convert_single_point(point) for point in input_data]

    def _convert_single_point(self, point):
        if len(point) == 3:
            z, y, x = point
            return (z * self.voxel_size[0], y * self.voxel_size[1], x * self.voxel_size[2])
        elif len(point) == 2:
            y, x = point
            return (0.0, y * self.voxel_size[1], x * self.voxel_size[2])
        return (0.0, 0.0, 0.0)

    def _save_intermediate_results(self, segments_bp, segments_connected, output_dir):
        """Save intermediate results (File IO)"""
        if not output_dir: return
        os.makedirs(output_dir, exist_ok=True)

        if self.is_3d:
            shape = self.skeleton.shape
        else:
            shape = self.skeleton.shape[1:]

        img_bp = self._fill_segments_channel(shape, segments_bp)
        if self.full_image_mode:
            path_seg = self.img_info.pipeline_paths.get('im_reconstruct_segments',
                                                        os.path.join(output_dir, 'im_reconstruct_segments.tif'))
        else:
            path_seg = os.path.join(output_dir, f'im_reconstruct_segments_object{self.object_id}.tif')
        tifffile.imwrite(path_seg, img_bp)

        if segments_connected is not None:
            img_connected = self._fill_segments_channel(shape, segments_connected)
            if self.full_image_mode:
                path_recon = self.img_info.pipeline_paths.get('im_reconstruct_segments_reconstructed',
                                                              os.path.join(output_dir,
                                                                           'im_reconstruct_segments_reconstructed.tif'))
                path_skel = self.img_info.pipeline_paths.get('im_reconstruct_skeleton',
                                                             os.path.join(output_dir, 'im_reconstruct_skeleton.tif'))
            else:
                path_recon = os.path.join(output_dir,
                                          f'im_reconstruct_segments_reconstructed_object{self.object_id}.tif')
                path_skel = os.path.join(output_dir, f'im_reconstruct_skeleton_object{self.object_id}.tif')

            skeleton_mask = (img_connected > 0).astype(np.uint8) * 255
            tifffile.imwrite(path_recon, img_connected)
            tifffile.imwrite(path_skel, skeleton_mask)

    def _fill_segments_channel(self, shape, segments):
        image = np.zeros(shape, dtype=np.uint16)
        for i, seg in enumerate(segments):
            for point in seg['path']:
                if self.is_3d:
                    z, y, x = point[:3]
                    if 0 <= z < image.shape[0] and 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                        image[z, y, x] = i + 1
                else:
                    z, y, x = point if len(point) == 3 else (0, point[0], point[1])
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        image[y, x] = i + 1
        return image
