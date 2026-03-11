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
    """Handle network visualization and report generation"""

    def __init__(self, self_info):
        # Store reference to main analyzer instance
        self.analyzer = self_info.analyzer
        self.info = self_info

    def visualize_3d_paths(self, output_dir, obj_id):
        """Draw complete 3D network paths"""
        if not self.analyzer.is_3d or len(self.analyzer.edge_properties) == 0:
            return

        try:
            # Create 3D figure
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Set background color
            fig.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')

            base_node_size = 15

            # Collect capacity values of all edges to create a global color map
            capacities = []
            for (u, v), properties_list in self.analyzer.edge_properties.items():
                for props in properties_list:
                    if 'path' in props and 'capacity' in props:
                        capacity = props['capacity']
                        if np.isfinite(capacity) and capacity > 0:
                            capacities.append(capacity)

            # Create color map (normalize using data from all edges)
            if capacities:
                # Use actual data range instead of percentiles
                vmin = min(capacities)
                vmax = max(capacities)

                # If the data range is too small, appropriately expand the range to ensure color variation is visible
                if vmax - vmin < 1e-5:  # All values are almost the same
                    vmin = vmin * 0.9 if vmin > 0 else 0
                    vmax = vmax * 1.1 if vmax > 0 else 1.0

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap('viridis')
                mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                mapper.set_array([])  # Must set empty array for colorbar to use
            else:
                print("Warning: No valid transport capacity values available for color mapping")
                mapper = None

            # Draw all edge paths
            for (u, v), properties_list in self.analyzer.edge_properties.items():
                for props in properties_list:
                    if 'path' not in props:
                        continue

                    # Get path and convert to pixel coordinates
                    path = props['path']
                    path_pixels = self._physical_to_pixel_coordinates(path)

                    # Ensure there are valid coordinate points
                    if not path_pixels:
                        continue

                    zs, ys, xs = zip(*[(p, p, p) for p in path_pixels])

                    # Calculate color
                    if mapper and 'capacity' in props:
                        capacity = props['capacity']
                        # Limit to valid range
                        if capacity < vmin: capacity = vmin
                        if capacity > vmax: capacity = vmax
                        color = mapper.to_rgba(capacity)
                    else:
                        color = 'cyan'  # Default color

                    # Draw path
                    ax.plot(xs, ys, zs,
                            color=color,
                            linewidth=2,  # Fixed line width of 2
                            alpha=0.7,
                            solid_capstyle='round'
                            )


            # Collect all node positions and draw them uniformly (keep unchanged)
            xs_n, ys_n, zs_n = [], [], []
            node_sizes = []
            for node_id, node_data in self.analyzer.graph.nodes(data=True):
                if 'pos' in node_data:
                    pixel_coord = self._physical_to_pixel_coordinates([node_data['pos']])
                    if pixel_coord:
                        z, y, x = pixel_coord
                        xs_n.append(x)
                        ys_n.append(y)
                        zs_n.append(z)
                        degree = self.analyzer.graph.degree[node_id]
                        size = max(5, min(30, base_node_size + degree * 1.5))
                        node_sizes.append(size)

            # Uniformly draw all nodes (keep unchanged)
            if xs_n and ys_n and zs_n and node_sizes:
                ax.scatter(xs_n, ys_n, zs_n,
                           s=node_sizes,
                           c='lime',
                           ec='white',
                           alpha=0.7)

            # Set axis labels (keep unchanged)
            unit_x = "X (μm)" if self.analyzer.voxel_size != 1 else "X (px)"
            unit_y = "Y (μm)" if self.analyzer.voxel_size != 1 else "Y (px)"
            unit_z = "Z (μm)" if self.analyzer.voxel_size != 1 else "Z (px)"

            ax.set_xlabel(unit_x)
            ax.set_ylabel(unit_y)
            ax.set_zlabel(unit_z)

            # Set viewing angle
            ax.view_init(elev=30, azim=45)

            # Title
            ax.set_title(f"Object {obj_id} 3D Network Paths", color='white', fontsize=14)

            # Save image
            output_path = os.path.join(output_dir, f"object_{obj_id}_3d_paths.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"3D path visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    def visualize_network_projection(self, output_dir, obj_id, bg_img=None):
        """Draw network path on 2D projection (XY projection)"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Set background
        if bg_img is not None:
            ax.imshow(bg_img, cmap='gray', origin='upper')
        else:
            if self.analyzer.is_3d:
                bg_img = np.max(self.analyzer.skeleton, axis=0)
            else:
                bg_img = self.analyzer.skeleton
            ax.imshow(bg_img, cmap='gray', origin='upper')

        # Define reasonable node size
        base_node_size = 10

        # Collect capacity values of all edges to create a global color map
        capacities = []
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' in props and 'capacity' in props:
                    capacity = props['capacity']
                    if np.isfinite(capacity) and capacity > 0:
                        capacities.append(capacity)

        # Create color map (normalize using data from all edges)
        if capacities:
            # Use actual data range instead of percentiles
            vmin = min(capacities)
            vmax = max(capacities)

            # If the data range is too small, appropriately expand the range to ensure color variation is visible
            if vmax - vmin < 1e-5:  # All values are almost the same
                vmin = vmin * 0.9 if vmin > 0 else 0
                vmax = vmax * 1.1 if vmax > 0 else 1.0

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('viridis')
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])  # Must set empty array for colorbar to use
        else:
            print("Warning: No valid transport capacity values available for color mapping")
            mapper = None

        # Draw all edge paths
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' not in props:
                    continue

                path = props['path']
                xs, ys = [], []

                # Convert path from physical coordinates to pixel coordinates
                path_pixels = self._physical_to_pixel_coordinates(path)

                # Ensure there are valid coordinate points
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

                # Ensure there are enough data points
                if len(xs) < 2:
                    continue

                # Calculate color
                if mapper and 'capacity' in props:
                    capacity = props['capacity']
                    # Limit to valid range
                    if capacity < vmin: capacity = vmin
                    if capacity > vmax: capacity = vmax
                    color = mapper.to_rgba(capacity)
                else:
                    color = 'cyan'  # Default color

                # Draw path
                ax.plot(xs, ys,
                        color=color,
                        linewidth=2,
                        alpha=0.7,
                        solid_capstyle='round'
                        )

        # Collect all node positions and sizes
        xs, ys = [], []
        node_sizes = []
        for node_id, node_data in self.analyzer.graph.nodes(data=True):
            if 'pos' in node_data:
                pixel_coords = self._physical_to_pixel_coordinates([node_data['pos']])
                if pixel_coords:
                    coord = pixel_coords
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

        # Uniformly draw all nodes
        if xs and ys and node_sizes:
            ax.scatter(
                xs, ys,
                s=node_sizes,
                c='lime',
                ec='white',
                alpha=0.7,
                zorder=5
            )

        # Set labels
        unit = " (px)"
        if any(v != 1 for v in self.analyzer.voxel_size):
            unit = " (μm)"

        ax.set_xlabel(f"X{unit}")
        ax.set_ylabel(f"Y{unit}")
        ax.set_title(f"Object {obj_id} Network Projection", fontsize=12)

        # Save image
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
                    z / voxel_size if voxel_size > 0 else 0,
                    y / voxel_size if voxel_size > 0 else 0,
                    x / voxel_size if voxel_size > 0 else 0
                ))
            elif len(point) == 2:
                y, x = point
                pixel_points.append((
                    0,
                    y / voxel_size if voxel_size > 0 else 0,
                    x / voxel_size if voxel_size > 0 else 0
                ))
        return pixel_points

    def export_paths_as_tiff(self, output_path, obj_id):
        """Export reconstructed paths as 3D TIFF image"""
        if not self.analyzer.is_3d: return
        path_image = np.zeros_like(self.analyzer.skeleton, dtype=np.uint16)
        path_id = 1
        for (u, v), properties_list in self.analyzer.edge_properties.items():
            for props in properties_list:
                if 'path' in props:
                    for point in props['path']:
                        z, y, x = map(int, point[:3])
                        if 0 <= z < path_image.shape and 0 <= y < path_image.shape and 0 <= x < path_image.shape:
                            path_image[z, y, x] = path_id
                path_id += 1
        save_path = os.path.join(output_path, f"object_{obj_id}_reconstructed_paths.tiff")
        io.imsave(save_path, path_image, check_contrast=False)
        print(f"Path reconstruction image saved to: {save_path}")

    # =========================================================================
    #  [New] Network Construction Visualization (Migrated and refactored from Analyzer)
    # =========================================================================

    def visualize_network_construction(self, segments_bp, segments_connected):
        """
        Visualize key steps of network construction
        Parameters come from return value of analyzer.analyze_network
        """
        # Recalculate temporary connected domains for visualization (keep Analyzer pure, do not store non-core states)
        connectivity = 3 if self.analyzer.is_3d else 2
        labeled = measure.label(self.analyzer.skeleton, connectivity=connectivity)

        if self.analyzer.full_image_mode:
            object_visualize_path = self.analyzer.img_info.graph_dir
        else:
            object_visualize_path = os.path.join(self.analyzer.img_info.graph_dir,
                                                 "object_" + str(self.analyzer.object_id))

        os.makedirs(object_visualize_path, exist_ok=True)

        # 1. Visualize network graph structure (Node-Edge Graph)
        graph_output_path = os.path.join(object_visualize_path, "network_graph.png")
        self.visualize_network_graph(output_path=graph_output_path)


    def visualize_network_graph(self, output_path=None):
        """Visualize network graph structure (NetworkX Graph)"""
        if not self.analyzer.graph:
            print("Network graph has not been built yet, cannot visualize")
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

    # --- Internal auxiliary plotting methods ---

    def _visualize_2d_network_graph(self):
        graph = self.analyzer.graph
        pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
        node_colors = self._get_node_colors(graph)

        nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=node_colors, alpha=0.8)
        self._draw_graph_edges(graph, pos)
        self._add_graph_legend()

    def _visualize_3d_network_graph(self):
        # Visualize 3D projection onto 2D plane
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
            coords = [(p, p) for p in path]  # Assuming 3D or 2D are in the last two dimensions
            for y, x in coords:
                if 0 <= y < vis_img.shape and 0 <= x < vis_img.shape:
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
            vis_img = labeled_skeleton if labeled_skeleton.ndim == 3 else labeled_skeleton

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
            vis_img = pixel_class if pixel_class.ndim == 3 else pixel_class

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
    Cytoskeleton Visualization Manager

    Parameters:
    analyzer (CytoskeletonAnalyzer3D): Cytoskeleton analyzer instance
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.img_info = analyzer.img_info
        self.is_3d = analyzer.is_3d
        self.voxel_size = analyzer.voxel_size
        self.network_cache = analyzer.network_cache
        self.feature_table = analyzer.feature_table

    def visualize_global_network(self, edge_attribute='length', cmap='viridis'):
        """Draw overall network projection of all objects - optimized version (uses edge_properties)"""
        # Create 2-row 1-column layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 24))

        # Set figure and subplot 2 background to black
        fig.patch.set_facecolor('black')
        ax2.set_facecolor('black')

        # Get background image
        if self.is_3d:
            raw_bg = np.max(self.analyzer.raw_image, axis=0)
        else:
            raw_bg = self.analyzer.raw_image

        # Max-min normalize original image
        raw_min = np.min(raw_bg)
        raw_max = np.max(raw_bg)
        if raw_max > raw_min:
            raw_bg = (raw_bg - raw_min) / (raw_max - raw_min)
        else:
            raw_bg = exposure.rescale_intensity(raw_bg, out_range=(0, 1))

        # Subplot 1: Show normalized original image (no foreground)
        im1 = ax1.imshow(raw_bg, cmap='gray', origin='upper', vmin=0, vmax=1)
        ax1.set_title("Normalized Original Image", fontsize=25, color='white')
        ax1.set_xlabel("X (μm)" if self.analyzer.apply_anisotropic else "X (px)")
        ax1.set_ylabel("Y (μm)" if self.analyzer.apply_anisotropic else "Y (px)")
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        # Prepare colormap data
        all_values = []  # Collect property values for all edges

        # First pass: collect property values of all edges
        for obj_id, analyzer in self.network_cache.items():
            if not hasattr(analyzer, 'edge_properties'):
                continue

            for edge_key, properties_list in analyzer.edge_properties.items():
                for props in properties_list:
                    value = props.get(edge_attribute, 0)
                    all_values.append(value)

        # Enhanced color mapping processing
        if all_values:
            # Define range using percentiles, avoid outlier influence
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)

            # Ensure vmax > vmin
            if vmax <= vmin:
                # Handling when all values are almost identical
                if vmin > 0:
                    vmax = vmin * 1.1
                else:
                    vmin, vmax = 1e-6, 1.0

            # Create normalizer and mapper
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])
        else:
            print("Warning: No valid transport capacity values available for color mapping")
            norm = None
            mapper = None

        # Second pass: draw paths (only in subplot 2)
        for obj_id, analyzer in self.network_cache.items():
            if not hasattr(analyzer, 'edge_properties'):
                continue

            # Iterate through all edges and fragments
            for edge_key, properties_list in analyzer.edge_properties.items():
                for props in properties_list:
                    # Skip edges with no path data
                    if 'path' not in props or len(props['path']) < 2:
                        continue

                    # Get property value
                    value = props.get(edge_attribute, 0)
                    path = props['path']

                    # Prepare coordinate lists
                    xs, ys = [], []

                    # Process path points
                    for point in path:
                        if self.is_3d:
                            # 3D point: (z,y,x) -> project to (y,x)
                            _, y, x = point[:3]
                        else:
                            # 2D point: may be (y,x) or (z,y,x) format
                            if len(point) == 3:  # (z,y,x) format but z=0
                                _, y, x = point
                            else:  # (y,x) format
                                y, x = point
                        xs.append(x)
                        ys.append(y)

                    # Ensure enough data points
                    if len(xs) < 2:
                        continue

                    # Calculate color
                    if mapper:
                        color = mapper.to_rgba(value)
                        # Fixed line width at 1.0 (no longer dynamic)
                        linewidth = 1.0
                    else:
                        color = 'cyan'
                        linewidth = 1.0

                    # Draw network path in subplot 2
                    ax2.plot(xs, ys,
                             color=color,
                             linewidth=linewidth,
                             alpha=0.7,
                             solid_capstyle='round')

        # Set labels and title for subplot 2
        ax2.set_title("Global Network Projection", fontsize=25, color='white')
        ax2.set_xlabel("X (μm)" if self.analyzer.apply_anisotropic else "X (px)", color='white')
        ax2.set_ylabel("Y (μm)" if self.analyzer.apply_anisotropic else "Y (px)", color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')

        # Uniform ratio and range for both subplots
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_aspect(ax1.get_aspect())

        # # Only invert Y-axis for subplot 1
        # ax1.invert_yaxis()

        # Fix: Add separate colorbar for subplot 2
        if mapper:
            # Get position info of subplot 2
            ax2_pos = ax2.get_position()

            # Create colorbar axis to the right of subplot 2
            cax = fig.add_axes([ax2_pos.x1 + 0.02, ax2_pos.y0, 0.02, ax2_pos.height])
            cbar = fig.colorbar(mapper, cax=cax, orientation='vertical')
            cbar.set_label(edge_attribute.upper(), fontsize=12, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Adjust layout to prevent overlapping colorbars
        plt.tight_layout()

        # Save image
        save_path = os.path.join(self.img_info.graph_dir, "global_network_projection.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig)


    def visualize_global_3d_paths(self, edge_attribute='capacity'):
        """Draw overall 3D network paths for all objects - modified version, colored by capacity"""
        if not self.is_3d:
            return

        try:
            fig = plt.figure(figsize=(18, 16))
            ax = fig.add_subplot(111, projection='3d')

            # Collect capacity values of all edges
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

            # Handle empty datasets
            if not all_capacities:
                print("Warning: No valid transport capacity values detected")
                return

            # Determine normalization range (use percentiles to avoid outlier influence)
            valid_capacities = [c for c in all_capacities if c > 0]
            if not valid_capacities:
                print("Warning: Transport capacity values for all edges are 0 or negative")
                return

            vmin = np.percentile(valid_capacities, 5)
            vmax = np.percentile(valid_capacities, 95)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array([])

            # Iterate over all objects and paths
            for obj_id, analyzer in self.network_cache.items():
                if not hasattr(analyzer, 'edge_properties'):
                    continue

                # Iterate over all edges and fragments
                for edge_key, properties_list in analyzer.edge_properties.items():
                    for props in properties_list:
                        if 'path' not in props:
                            continue

                        path = props['path']
                        capacity = props.get(edge_attribute, 0)
                        color = mapper.to_rgba(capacity)  # Get color using current capacity value

                        # Extract coordinates
                        xs, ys, zs = [], [], []
                        for point in path:
                            if len(point) >= 3:  # Ensure it is a 3D point
                                z, y, x = point[:3]
                                xs.append(x)
                                ys.append(y)
                                zs.append(z)

                        if xs and ys and zs:  # Ensure valid points exist
                            # Draw path
                            ax.plot(xs, ys, zs,
                                    color=color,
                                    linewidth=0.8 + 3 * (capacity / vmax),
                                    alpha=0.7,
                                    solid_capstyle='round')

            # Add colorbar
            cbar = fig.colorbar(mapper, ax=ax, shrink=0.8, pad=0.1)
            cbar.set_label('Transport Capacity', fontsize=12)
            cbar.formatter = ScalarFormatter(useMathText=False)
            cbar.update_ticks()

            # Set axes
            ax.set_xlabel("X (μm)" if self.voxel_size != 1 else "X (px)")
            ax.set_ylabel("Y (μm)" if self.voxel_size != 1 else "Y (px)")
            ax.set_zlabel("Z (μm)" if self.voxel_size != 1 else "Z (px)")

            # Set view
            ax.view_init(elev=25, azim=35)

            # Title
            ax.set_title("Global 3D Network Paths (Capacity Colored)", fontsize=16)

            # Save image
            save_path = os.path.join(self.img_info.graph_dir, "global_3d_network_paths.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Global 3D network visualization failed: {str(e)}")
            traceback.print_exc()
            plt.close('all')

    def visualize_network(self,
                          edge_attribute='capacity',
                          background_type='raw',
                          cmap='viridis',  # Retain color parameter
                          output_suffix=None):
        """Network visualization method (Improved version)

        Args:
            edge_attribute: Edge attribute for coloring
            background_type: Background type (raw/annotated)
            cmap: Color map table (default viridis)
            output_suffix: Output filename suffix
        """
        # Get and preprocess background
        bg_img = self._get_background_image(background_type)
        bg_img = (bg_img - bg_img.min()) / (bg_img.max() - bg_img.min())
        # bg_img = exposure.rescale_intensity(bg_img, out_range=(0.1, 0.9))  # Smart contrast adjustment

        # Create canvas
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(bg_img, cmap='gray', vmin=0.1, vmax=0.9)

        # Edge data collection (with validation)
        all_edges = []
        for obj in self.feature_table:
            # Safely get object ID
            obj_id = obj.get('object_id')
            if not obj_id:
                continue

            # Validate analyzer validity
            analyzer = self.network_cache.get(obj_id)
            if not analyzer or not hasattr(analyzer, 'graph'):
                continue

            # Generate node coordinates (filter invalid values)
            node_pos = {}
            for n, coord in enumerate(analyzer.centroids):
                try:
                    if self.is_3d:  # 3D image processing
                        z, y, x = map(float, coord)
                        if all(np.isfinite([z, y, x])):
                            node_pos[n] = (
                                y * self.voxel_size,
                                x * self.voxel_size
                            )
                    else:  # 2D image processing
                        # Handle possible (z,y,x) or (y,x) formats
                        if len(coord) == 3:  # (z,y,x) format but z=0
                            _, y, x = coord
                        else:  # (y,x) format
                            y, x = coord

                        if all(np.isfinite([y, x])):
                            node_pos[n] = (
                                y * self.voxel_size,
                                x * self.voxel_size
                            )
                except (TypeError, ValueError) as e:
                    print(f"Exception processing node {n} coordinates: {str(e)}")
                    continue

            # Collect edge data
            for u, v, data in analyzer.graph.edges(data=True):
                if u in node_pos and v in node_pos:
                    y0, x0 = node_pos[u]
                    y1, x1 = node_pos[v]
                    all_edges.append((
                        y0, x0, y1, x1,
                        data.get(edge_attribute, 0)
                    ))

        # Visualize edges (with normalization)
        if all_edges:
            values = [e for e in all_edges]
            valid_values = [v for v in values if v > 0]

            if valid_values:
                # Dynamic range calculation
                vmin = np.percentile(valid_values, 5)
                vmax = np.percentile(valid_values, 95)
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

                # Draw segments
                for y0, x0, y1, x1, val in all_edges:
                    if val <= 0:
                        continue
                    ax.plot(
                        [x0, x1], [y0, y1],  # Note XY coordinate conversion
                        color=mapper.to_rgba(val),
                        linewidth=0.3 + 2 * (val / vmax),  # Dynamic line width
                        alpha=0.7,
                        solid_capstyle='round'
                    )

                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(mapper, cax=cax, label=edge_attribute.upper())

        # Output configuration
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
        """Visualize complete edge paths (instead of endpoint lines) - New feature

        Args:
            background_type: Background type (raw/ridge/skeleton)
            edge_attribute: Edge attribute for coloring (e.g., 'capacity'/'length')
            cmap: Color map table
            output_suffix: Output filename suffix
        """
        # Get and preprocess background
        bg_img = self._get_background_image(background_type)
        bg_img = exposure.rescale_intensity(bg_img, out_range=(0.1, 0.9))

        # Create canvas
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(bg_img, cmap='gray', vmin=0.1, vmax=0.9)

        # Collect all edge paths and attribute values
        all_paths = []
        edge_values = []

        # Iterate over all objects and network analyzers
        for obj_id, analyzer in self.network_cache.items():
            # Check if analyzer contains path info
            if not hasattr(analyzer, 'edge_properties'):
                continue

            # Iterate over all edge paths
            for edge_key, props in analyzer.edge_properties.items():
                # Skip edges without path data
                if 'path' not in props or len(props['path']) < 2:
                    continue

                # Get property value
                value = props.get(edge_attribute, 0)
                if edge_attribute not in props:
                    # Attempt to get attribute from graph edge
                    try:
                        graph_edge = analyzer.graph.edges[edge_key]
                        value = graph_edge.get(edge_attribute, 0)
                    except:
                        value = 0

                edge_values.append(value)
                all_paths.append(props['path'])

        # If no valid paths, save and return directly
        if not all_paths:
            ax.set_title("No Paths Found")
            ax.axis('off')
            save_path = os.path.join(self.img_info.graph_dir, f"no_paths_warning.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Normalize property values for color mapping
        valid_values = [v for v in edge_values if v > 0]
        if valid_values:
            vmin = np.percentile(valid_values, 5)
            vmax = np.percentile(valid_values, 95)
        else:
            vmin, vmax = 0, 1

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Draw path segment by segment
        for path_idx, path_points in enumerate(all_paths):
            value = edge_values[path_idx]
            color = mapper.to_rgba(value) if valid_values else 'cyan'

            # Draw path in segments
            for i in range(1, len(path_points)):
                start = path_points[i - 1]
                end = path_points[i]

                # Handle coordinates of different dimensions
                if len(start) == 3:  # 3D point:(z,y,x)
                    y0, x0 = start, start
                else:  # 2D point:(y,x)
                    y0, x0 = start, start if len(start) > 1 else (0, 0)

                if len(end) == 3:  # 3D point:(z,y,x)
                    y1, x1 = end, end
                else:  # 2D point:(y,x)
                    y1, x1 = end, end if len(end) > 1 else (0, 0)

                ax.plot([x0, x1], [y0, y1],  # Convert XY order
                        color=color,
                        linewidth=0.3 + 2 * (value / vmax if vmax > 0 else 0.3),
                        alpha=0.7,
                        solid_capstyle='round')

        # Add colorbar (if valid attribute values exist)
        if valid_values:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(mapper, cax=cax, label=edge_attribute.upper())

        # Output configuration
        filename = f"paths_{background_type}"
        if output_suffix:
            filename += f"_{output_suffix}"
        save_path = os.path.join(self.img_info.graph_dir, f"{filename}.png")

        ax.set_title(f"Full Path Visualization ({edge_attribute})")
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _get_background_image(self, bg_type):
        """Get background image"""
        if self.is_3d:
            # Use max projection for 3D
            if bg_type == 'raw':
                bg_img = np.max(self.analyzer.raw_image, axis=0)
            elif bg_type == 'ridge':
                bg_img = np.max(self.analyzer.ridge_contrast_image, axis=0)
            elif bg_type == 'skeleton':
                bg_img = np.max(self.analyzer.skeleton, axis=0)
        else:
            # Use raw image directly for 2D
            if bg_type == 'raw':
                bg_img = self.analyzer.raw_image
            elif bg_type == 'ridge':
                bg_img = self.analyzer.ridge_contrast_image
            elif bg_type == 'skeleton':
                bg_img = self.analyzer.skeleton
        return bg_img

    def _select_cmap(self, bg_img):
        """Smart selection of background color map"""
        return 'gray_r' if np.median(bg_img) < 128 else 'gray'