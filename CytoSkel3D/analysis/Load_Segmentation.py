import os
import numpy as np
from warnings import warn
from scipy import ndimage, stats
from skimage import io, measure, morphology, feature, graph, img_as_ubyte, exposure
from tifffile import tifffile


class Load_Segmentation:

    def __init__(self, img_info, skeleton, params=None):
        self.img_info = img_info
        self.params = params
        self.skeleton = skeleton
        self.full_image_mode = params.get('full_image_mode', False)

    def _init_mask_system(self):
        """Initialize mask system - modified logic"""
        # Case 1: Full image mode
        if self.full_image_mode:
            object_mask = np.ones_like(self.skeleton, dtype=bool)
            labeled_objects = np.ones(self.skeleton.shape, dtype=np.uint16)
            print("Full image mode: using the entire image as a single object")
            return object_mask, labeled_objects

        # Case 2: External mask available
        elif self._has_external_mask():
            object_mask = self._process_external_mask()
            labeled_objects = object_mask
            print("Using external mask")
            return object_mask, labeled_objects

        # Case 3: No external mask - creating convex hull mask based on the entire skeleton
        else:
            print("No external mask: creating convex hull mask based on the entire skeleton")

            # Initialize all-zero mask
            object_mask = np.zeros_like(self.skeleton, dtype=bool)
            labeled_objects = np.zeros(self.skeleton.shape, dtype=np.uint16)

            # Create convex hull layer by layer
            for z in range(self.skeleton.shape):
                layer = self.skeleton[z]

                # If the layer has any skeleton points
                if np.any(layer):
                    # Create convex hull for the layer

                    convex_layer = morphology.convex_hull_image(layer)
                    object_mask[z] = convex_layer

            # Mark the entire convex hull region as a single object
            labeled_objects[object_mask] = 1

            # Save mask file
            mask_save_path = self.img_info.pipeline_paths['mask']
            tifffile.imwrite(
                mask_save_path,
                labeled_objects,
                bigtiff=True,
            )

            # output_path = "D:/Data/KerNet_data/tmp.tiff"
            # # Create RGB image for visualization
            # output_img = np.zeros((*labeled_objects.shape, 3), dtype=np.uint8)
            #
            # # # Original skeleton displayed in green
            # # skeleton_rgb = self.skeleton.astype(bool)
            # # output_img[skeleton_rgb] =
            # #
            # # # Convex hull displayed in red (boundary only)
            # # hull_boundary = labeled_objects & ~morphology.binary_erosion(labeled_objects)
            # # output_img[hull_boundary] =
            #
            # # Save 3D TIFF stack
            # io.imsave(output_path, labeled_objects, plugin='tifffile')

            return object_mask, labeled_objects

    def _has_external_mask(self):
        """Check for external mask"""
        # Check if the 'maskpath' attribute exists, is not None, and is an existing file
        return (hasattr(self.img_info, 'maskpath') and
                self.img_info.maskpath is not None and
                os.path.isfile(self.img_info.maskpath))

    def _process_external_mask(self):
        """Process external multi-label mask (retain original integer values)"""
        # Read using skimage to support multiple formats
        mask = io.imread(self.img_info.maskpath)

        # Get all non-zero unique values and sort in ascending order
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values != 0]
        unique_values.sort()

        # Create a new mask and remap labels
        new_mask = np.zeros_like(mask)
        for new_id, orig_val in enumerate(unique_values, 1):
            new_mask[mask == orig_val] = new_id

        print(f"Unique values of the mask after remapping: {np.unique(new_mask)}")  # Should now be [0,1,2,...]

        # Verify that the data type is integer
        if not np.issubdtype(new_mask.dtype, np.integer):
            raise TypeError("External mask must be an integer type (different objects represented by different values)")

        # Dimension check (maintain ZYX order)
        if new_mask.shape != self.skeleton.shape:
            # Example logic for automatic resizing (needs to be implemented based on specific data conditions)
            # For example, center crop or interpolation scaling, modify here according to actual conditions
            new_mask = self._resize_3d_mask(new_mask, target_shape=self.skeleton.shape)
            warn(f"Automatically resizing external mask to {self.skeleton.shape}")

        # Remove small objects (optional, based on requirements)
        cleaned = morphology.remove_small_objects(new_mask, min_size=self.params['min_object_size'])

        return cleaned

    def _resize_3d_mask(self, mask, target_shape):
        """3D mask resizing (Example: Nearest-neighbor interpolation to maintain integer labels)"""
        from scipy.ndimage import zoom
        zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
        resized = zoom(mask.astype(float), zoom_factors, order=0)  # order=0 nearest-neighbor interpolation
        return resized.astype(mask.dtype)

    def _generate_hybrid_mask(self):
        """Generate mask using hybrid strategy: layer-by-layer processing -> 3D integration"""
        # Generate base mask layer by layer
        layered_masks = self._generate_layered_masks()

        # 3D integration
        stacked_mask = np.stack(layered_masks, axis=0)
        return self._refine_mask_3d(stacked_mask)

    def _generate_layered_masks(self):
        """Generate optimized masks layer by layer (core modification)"""
        masks = []
        for z in range(self.binary_image.shape):
            layer = self.binary_image[z]

            # Layer-by-layer processing pipeline
            processed = self._process_single_layer(layer)
            masks.append(processed)

        return masks

    def _process_single_layer(self, layer):
        """Single layer processing pipeline (2D optimization)"""
        # 1. Morphological optimization
        closed = morphology.binary_closing(layer, morphology.disk(1))

        # 2. Region filtering
        cleaned = self._filter_layer_regions(closed)

        # 3. Hole filling

        filled = ndimage.binary_fill_holes(cleaned)

        return filled

    def _filter_layer_regions(self, layer):
        """Single layer region filtering"""
        labels = measure.label(layer)
        regions = measure.regionprops(labels)

        valid_mask = np.zeros_like(layer)
        min_area = self.params.get('min_layer_area', 10)  # Minimum number of pixels per layer

        for reg in regions:
            if reg.area >= min_area and not self._touches_layer_border(reg, layer.shape):
                valid_mask[reg.coords[:, 0], reg.coords[:, 1]] = 1

        return valid_mask

    @staticmethod
    def _touches_layer_border(region, shape):
        """Check if it touches the single layer border"""
        return (region.bbox == 0 or region.bbox == shape or
                region.bbox == 0 or region.bbox == shape)

    def _refine_mask_3d(self, mask_3d):
        # Step adjustment:
        # 1. First fill holes
        filled = ndimage.binary_fill_holes(mask_3d)
        # 2. Connected component processing
        labels = measure.label(filled)
        min_vol = self.params['min_object_size']
        cleaned = morphology.remove_small_objects(labels, min_size=min_vol)
        # 3. Border check
        border_cleaned = self._remove_crosslayer_border_objects(cleaned)
        # 4. Final closing operation
        return morphology.binary_closing(border_cleaned, footprint=morphology.ball(1))

    def _remove_crosslayer_border_objects(self, labeled_3d):
        """Improved multi-label border check"""
        border_mask = np.zeros(labeled_3d.shape, dtype=bool)
        border_mask[:, 1:-1, 1:-1] = True  # Keep entirely on the Z axis, only check XY borders

        for obj_id in np.unique(labeled_3d)[1:]:
            obj_mask = labeled_3d == obj_id
            if np.any(obj_mask & ~border_mask):
                labeled_3d[obj_mask] = 0  # Delete objects touching XY borders

        return labeled_3d