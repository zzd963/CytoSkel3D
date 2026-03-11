import numpy as np
import pandas as pd
import os
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev, CubicSpline
from skimage.morphology import dilation, disk, ball, erosion
from skimage import filters
from tqdm import tqdm


def _convert_numpy_to_python(obj):
    """Recursively convert NumPy types to native Python types for JSON/NPY serialization"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_python(v) for v in obj)
    return obj


class SyntheticFilamentGenerator:
    def __init__(self, img_size=(512, 512), z_slices=5, save_params=True):
        """
        Synthetic actin filament network generator (Physics-enhanced version)
        """
        self.img_size = img_size
        self.z_slices = z_slices
        self.is_3d = z_slices > 1
        self.save_params = save_params
        self.params_df = pd.DataFrame()
        self.current_params = {}

    def _generate_single_filament(self, start_point=None, length_range=(30, 100), curvature=0.3, z_layer=None):
        """Generate a single actin filament curve"""
        # Random start point
        if start_point is None:
            start_point = (np.random.randint(0, self.img_size),
                           np.random.randint(0, self.img_size))

        # Random Z-layer assignment
        if self.is_3d and z_layer is None:
            z_layer = np.random.randint(0, self.z_slices)

        # Random length
        length = np.random.randint(*length_range)

        # Random end point direction
        angle = np.random.uniform(0, 2 * np.pi)
        end_point = (start_point + int(length * np.cos(angle)),
                     start_point + int(length * np.sin(angle)))

        # Control point generation
        ctrl_points = [start_point]
        for i in range(1, 3):
            dist = i * length / 3
            offset = np.random.normal(0, curvature * length, size=2)
            ctrl_points.append((
                int(start_point + dist * np.cos(angle) + offset),
                int(start_point + dist * np.sin(angle) + offset)
            ))
        ctrl_points.append(end_point)

        # Dynamic interpolation points: increase density to prevent 3D breakage
        # At least 2 points per pixel
        num_points = max(min(int(length * 3), 1000), 50)

        try:
            tck, u = splprep(np.array(ctrl_points).T, s=0)
            points = np.array(splev(np.linspace(0, 1, num_points), tck)).T.astype(int)
        except:
            points = np.linspace(start_point, end_point, num_points).astype(int)

        # Crop to image bounds
        valid_points = []
        for point in points:
            if (0 <= point < self.img_size and
                    0 <= point < self.img_size):
                valid_points.append(point)

        return np.array(valid_points), z_layer, length, ctrl_points

    def _create_crossing(self, base_filament, base_z, cross_angle_range=(30, 90),
                         cross_length_range=(15, 100)):
        """Create crossing points"""
        if len(base_filament) < 20:
            return np.array([]), None, None, None, None

        cross_idx = np.random.randint(10, len(base_filament) - 10)
        cross_point = base_filament[cross_idx]

        direction = base_filament[cross_idx + 5] - base_filament[cross_idx - 5]
        base_angle = np.arctan2(direction, direction) * 180 / np.pi

        cross_angle = np.random.uniform(*cross_angle_range)
        actual_angle = base_angle + cross_angle
        angle_rad = np.deg2rad(actual_angle)

        crossing_type = np.random.choice(, p=[0.7, 0.3])

        if crossing_type == 3:
            length = np.random.randint(*cross_length_range)
            end_point = (
                int(cross_point + length * np.cos(angle_rad)),
                int(cross_point + length * np.sin(angle_rad))
            )
            num_points = max(min(int(length * 3), 500), 20)

            # Use linspace to ensure continuity
            x_vals = np.linspace(cross_point, end_point, num_points)
            y_vals = np.linspace(cross_point, end_point, num_points)
            cross_filament = np.column_stack((x_vals, y_vals)).astype(int)

        else:
            factor1 = np.random.uniform(0.1, 0.9)
            length1 = int(np.random.randint(*cross_length_range) * factor1)
            length2 = int(np.random.randint(*cross_length_range) * (1 - factor1))

            end_point1 = (
                int(cross_point + length1 * np.cos(angle_rad)),
                int(cross_point + length1 * np.sin(angle_rad))
            )
            end_point2 = (
                int(cross_point - length2 * np.cos(angle_rad)),
                int(cross_point - length2 * np.sin(angle_rad))
            )

            num_points1 = max(min(int(length1 * 3), 500), 20)
            num_points2 = max(min(int(length2 * 3), 500), 20)

            pts1 = np.column_stack((
                np.linspace(cross_point, end_point1, num_points1),
                np.linspace(cross_point, end_point1, num_points1)
            ))
            pts2 = np.column_stack((
                np.linspace(cross_point, end_point2, num_points2),
                np.linspace(cross_point, end_point2, num_points2)
            ))

            cross_filament = np.vstack((pts1, pts2)).astype(int)

        cross_z = base_z if self.is_3d else None
        return cross_filament, cross_point, actual_angle, cross_z, crossing_type

    def generate_filaments(self, num_filaments=10, crossing_prob=0.3,
                           curvature_range=(0.1, 0.5), cross_angle_range=(30, 90),
                           length_range=(30, 100), cross_length_range=(15, 100)):
        """Generate actin filament network"""
        if self.is_3d:
            base_image = np.zeros((self.z_slices, *self.img_size), dtype=np.uint8)
        else:
            base_image = np.zeros(self.img_size, dtype=np.uint8)

        filaments = []
        crossing_points = []

        filament_params = {
            'img_size': self.img_size,
            'z_slices': self.z_slices,
            'num_filaments': num_filaments,
            'crossing_prob': crossing_prob,
            'curvature_range': curvature_range,
            'cross_angle_range': cross_angle_range,
            'cross_length_range': cross_length_range,
            'length_range': length_range,
            'filaments': []
        }
        self.current_params.update(filament_params)

        for i in range(num_filaments):
            curvature = np.random.uniform(*curvature_range)
            filament, z_layer, length, ctrl_points = self._generate_single_filament(
                curvature=curvature, length_range=length_range
            )

            if len(filament) > 0:
                filaments.append((filament, z_layer, ctrl_points))

                self.current_params['filaments'].append({
                    'id': f"f{i}",
                    'type': 'base',
                    'points': filament.tolist(),
                    'curvature': float(curvature),
                    'length': int(length),
                    'z_layer': int(z_layer) if z_layer is not None else None,
                    'ctrl_points': [list(p) for p in ctrl_points]
                })

                if np.random.random() < crossing_prob and len(filament) > 20:
                    cross_filament, cross_point, cross_angle, cross_z, crossing_type = self._create_crossing(
                        filament, z_layer, cross_angle_range, cross_length_range
                    )

                    if len(cross_filament) > 0:
                        cross_ctrl_points = [tuple(cross_point)]
                        # Simplified cross control point recording
                        filaments.append((cross_filament, cross_z, cross_ctrl_points))
                        crossing_points.append(cross_point)

                        self.current_params['filaments'].append({
                            'id': f"c{len(filaments)}",
                            'type': 'cross',
                            'points': cross_filament.tolist(),
                            'cross_angle': float(cross_angle),
                            'crossing_type': int(crossing_type),
                            'base_filament_id': f"f{i}",
                            'z_layer': int(cross_z) if cross_z is not None else None,
                            'ctrl_points': []
                        })

        # Draw
        # Optimization: Use NumPy advanced indexing to improve speed
        for filament_data in filaments:
            filament = filament_data
            z_layer = filament_data

            # Boundary check
            valid_mask = (
                    (filament[:, 0] >= 0) & (filament[:, 0] < self.img_size) &
                    (filament[:, 1] >= 0) & (filament[:, 1] < self.img_size)
            )
            valid_pts = filament[valid_mask]

            if self.is_3d and z_layer is not None:
                if 0 <= z_layer < self.z_slices:
                    base_image[z_layer, valid_pts[:, 1], valid_pts[:, 0]] = 255
            else:
                base_image[valid_pts[:, 1], valid_pts[:, 0]] = 255

        self.current_params = _convert_numpy_to_python(self.current_params)
        return base_image, filaments, crossing_points

    def thicken_filaments(self, base_image, kernel_size_range=(3, 7)):
        """Thicken actin filaments (3D optimized version)"""
        if isinstance(kernel_size_range, tuple) and len(kernel_size_range) == 2:
            if kernel_size_range == kernel_size_range:
                kernel_size = kernel_size_range
            else:
                kernel_size = np.random.randint(*kernel_size_range)
        else:
            kernel_size = kernel_size_range

        self.current_params['thicken_kernel_size'] = int(kernel_size)

        if self.is_3d:
            # Key improvement: Use 3D spherical structuring element
            # Solve "coin stacking" artifacts, ensure tubular structures are continuous in 3D space
            radius = max(1, kernel_size // 2)
            kernel = ball(radius)
            # Note: For large images, 3D dilation might be slow
            thickened = dilation(base_image, kernel)
        else:
            kernel = disk(kernel_size)
            thickened = dilation(base_image, kernel)

        return thickened

    def add_blur(self, image, sigma_range=(1, 3)):
        """Add optical blur (Physically accurate anisotropic version)"""
        if isinstance(sigma_range, tuple):
            sigma = np.random.uniform(*sigma_range)
        else:
            sigma = sigma_range

        self.current_params['sigma'] = float(sigma)

        if self.is_3d:
            # Physical model: Z-axis PSF is typically 2.5-3.0 times the XY axis (due to diffraction limit)
            # We apply this blur on an isotropic grid, subsequent z_scale resampling will automatically handle sampling anisotropy
            psf_anisotropy = 3.0
            sigma_z = sigma * psf_anisotropy

            # Apply anisotropic Gaussian blur (sigma_z, sigma_y, sigma_x)
            blurred = gaussian_filter(image, sigma=(sigma_z, sigma, sigma))

            self.current_params['psf_anisotropy'] = psf_anisotropy
            self.current_params['sigma_z_physical'] = float(sigma_z)
        else:
            blurred = gaussian_filter(image, sigma=sigma)

        return blurred

    def add_noise(self, image, snr_range=(3, 7), random_seed=None):
        """
        Add noise conforming to bio-image paper standards (Revert to original V1 Gaussian noise logic)

        Original logic reproduction:
        1. Use Otsu thresholding to separate signal and background.
        2. Erode background mask to remove edges.
        3. Calculate mean of signal region.
        4. Calculate target noise standard deviation based on SNR (target_noise_std = mean_signal / snr).
        5. Generate and superimpose Gaussian noise.
        """
        # Create independent random generator
        rng = np.random.default_rng(random_seed)

        # Get SNR value
        if isinstance(snr_range, tuple):
            snr = np.random.uniform(*snr_range)
        else:
            snr = snr_range

        # Improved signal/background separation (logic ported from original code)
        if image.max() > 0:
            threshold = filters.threshold_otsu(image)
            signal_mask = image > threshold
        else:
            signal_mask = np.zeros_like(image, dtype=bool)

        # Create pure background mask (excluding signal edges)
        background_mask_ = ~signal_mask

        # Erosion operation to get pure background region
        if self.is_3d:
            # 3D image uses 3D morphological operations

            background_mask = erosion(background_mask_, ball(3))
        else:
            # 2D image uses disk structuring element
            background_mask = erosion(background_mask_, disk(3))

        # Calculate signal mean
        mean_signal = np.mean(image[signal_mask]) if np.any(signal_mask) else 100

        # Calculate required noise standard deviation (core formula)
        target_noise_std = mean_signal / snr

        # Generate Gaussian noise
        normalized = image.astype(float) / 255.0
        # Note: the scale parameter of rng.normal corresponds to standard deviation
        noise = rng.normal(0, target_noise_std / 255.0, size=image.shape)

        # Superimpose noise and clip
        noisy = np.clip(normalized + noise, 0, 1) * 255
        noisy = noisy.astype(np.uint8)

        # Calculate actual background standard deviation (for verification)
        actual_bg_std = np.std(noisy[background_mask]) if np.any(background_mask) else 0

        # Record parameters
        self.current_params['noise_snr'] = float(snr)
        self.current_params['target_noise_std'] = float(target_noise_std)
        self.current_params['actual_bg_std'] = float(actual_bg_std)
        self.current_params['mean_signal'] = float(mean_signal)
        self.current_params['noise_model'] = 'Signal-Dependent Gaussian (V1)'

        return noisy

    def _generate_z_curve(self, num_points, base_z, z_variation, derivative_range=(0.5, 2.0)):
        """Generate smooth Z-axis variation curve"""
        t = [0, 0.3, 0.7, 1.0]
        z_amplitude = min(5, z_variation / 2)
        z_base = base_z

        points = []
        for i in range(4):
            if i == 0 or i == 3:
                z_val = z_base + np.random.uniform(-z_amplitude, z_amplitude)
            else:
                z_val = z_base + np.random.uniform(-z_amplitude * 1.5, z_amplitude * 1.5)

            # Derivative control
            if i == 0 or i == 3:
                deriv = np.random.uniform(derivative_range, derivative_range) * np.sign(np.random.randn())
            else:
                if np.random.random() > 0.7:
                    deriv = 0
                else:
                    deriv = np.random.uniform(-derivative_range, derivative_range)
            points.append((z_val, deriv))

        cs = CubicSpline(t, [p for p in points],
                         bc_type=((1, points), (1, points)))

        u = np.linspace(0, 1, num_points)
        z_curve = cs(u)

        # Boundary constraints
        z_curve = np.clip(z_curve, 0, self.z_slices - 1)

        # Smoothness constraints
        for i in range(1, len(z_curve)):
            diff = z_curve[i] - z_curve[i - 1]
            if abs(diff) > 1.0:
                z_curve[i] = z_curve[i - 1] + np.sign(diff) * 1.0

        return z_curve

    def generate_3d_structure(self, base_2d_image, filaments, z_variation=12):
        """Generate 3D structure from 2D filament network (fix connectivity issues)"""
        self.current_params['z_variation'] = int(z_variation)
        base_3d_image = np.zeros((self.z_slices, *self.img_size), dtype=np.uint8)
        filaments_with_z = []

        for i, filament_data in enumerate(filaments):
            filament = filament_data
            base_z = filament_data
            num_points = len(filament)

            z_curve = self._generate_z_curve(num_points, base_z, z_variation)

            # Draw: Fill Z-axis breaks using linear interpolation
            for j in range(len(filament) - 1):
                p1 = filament[j]
                z1 = z_curve[j]
                p2 = filament[j + 1]
                z2 = z_curve[j + 1]

                # Calculate steps between two points to ensure coverage of each voxel
                dist = max(abs(p2 - p1), abs(p2 - p1), abs(z2 - z1))
                steps = int(np.ceil(dist * 1.5))  # 1.5x oversampling

                for k in range(steps + 1):
                    t = k / steps if steps > 0 else 0
                    curr_x = int(p1 + (p2 - p1) * t)
                    curr_y = int(p1 + (p2 - p1) * t)
                    curr_z = int(z1 + (z2 - z1) * t)

                    if (0 <= curr_x < self.img_size and
                            0 <= curr_y < self.img_size and
                            0 <= curr_z < self.z_slices):
                        base_3d_image[curr_z, curr_y, curr_x] = 255

            filaments_with_z.append((filament, z_curve))

            if i < len(self.current_params['filaments']):
                self.current_params['filaments'][i]['z_coords'] = list(z_curve.astype(float))

        return base_3d_image, filaments_with_z

    def apply_anisotropic_sampling(self, volume, z_scale):
        """Physically accurate anisotropic sampling"""
        if z_scale >= 1.0 or not self.is_3d:
            return volume

        original_z = volume.shape
        new_z = max(1, int(original_z * z_scale))
        sampled_volume = np.zeros((new_z, *volume.shape[1:]), dtype=volume.dtype)

        for z_idx in range(new_z):
            orig_z_pos = z_idx / z_scale if z_scale > 0 else 0  # Inverse mapping
            # Or simple linear mapping: z_idx / (new_z-1) * (original_z-1)
            # Using linear interpolation mapping is more robust:
            orig_z_pos = z_idx / (new_z - 1) * (original_z - 1) if new_z > 1 else 0

            z0 = int(np.floor(orig_z_pos))
            z1 = min(z0 + 1, original_z - 1)
            weight = orig_z_pos - z0

            if z1 < original_z:
                sampled_volume[z_idx] = (1 - weight) * volume[z0] + weight * volume[z1]
            else:
                sampled_volume[z_idx] = volume[z0]

        return sampled_volume

    def generate_synthetic_image(self, num_filaments=10, crossing_prob=0.3,
                                 curvature_range=(0.1, 0.5), cross_angle_range=(30, 90),
                                 kernel_size_range=(3, 7), sigma_range=(1, 3),
                                 snr_range=(3, 7), length_range=(30, 100),
                                 cross_length_range=(15, 100), z_variation=12, z_scale=1.0,
                                 noise_random_seed=None):
        """Generate complete synthetic image pipeline"""
        self.current_params['z_scale'] = float(z_scale)
        self.current_params['initial_z_slices'] = self.z_slices

        # 1. Generate 2D skeleton
        base_2d_img, filaments, cross_points = self.generate_filaments(
            num_filaments=num_filaments, crossing_prob=crossing_prob,
            curvature_range=curvature_range, cross_angle_range=cross_angle_range,
            length_range=length_range, cross_length_range=cross_length_range
        )

        if noise_random_seed is None:
            noise_random_seed = np.random.randint(0, 1000)
        self.current_params['noise_random_seed'] = int(noise_random_seed)

        # 2. Generate 3D structure (isotropic grid)
        if self.is_3d:
            base_img, self.filaments_with_z = self.generate_3d_structure(base_2d_img, filaments, z_variation)
        else:
            base_img = base_2d_img
            self.filaments_with_z = None

        # 3. Thicken (3D spherical dilation)
        thick_img = self.thicken_filaments(base_img, kernel_size_range)

        # 4. Blur (apply physical PSF anisotropy)
        # Note: operating on isotropic grid here, sigma_z ≈ 3 * sigma_xy
        blurred_img = self.add_blur(thick_img, sigma_range)

        # 5. Noise (revert to original Gaussian noise model)
        noisy_img = self.add_noise(blurred_img, snr_range, random_seed=noise_random_seed)

        # 6. Anisotropic sampling (Z-axis compression)
        # This step compresses physical blur and structure onto the low-resolution Z-axis
        if self.is_3d and z_scale < 1.0:
            final_img = self.apply_anisotropic_sampling(noisy_img, z_scale)
            base_img_final = self.apply_anisotropic_sampling(base_img, z_scale)
        else:
            final_img = noisy_img
            base_img_final = base_img

        return final_img, base_img_final, self.current_params

    def batch_generate(self, output_dir, num_images=10, **kwargs):
        """Batch generate"""
        os.makedirs(output_dir, exist_ok=True)
        all_params = []

        with tqdm(total=num_images, desc="Generating synthetic images") as pbar:
            for i in range(num_images):
                final_img, base_img, params = self.generate_synthetic_image(**kwargs)
                img_id = f"synth_{i:04d}"
                params['image_id'] = img_id

                params = _convert_numpy_to_python(params)

                # Save TIFF
                if self.is_3d:
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}.tiff"), final_img, imagej=True)
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}_base.tiff"), base_img, imagej=True)
                else:
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}.tiff"), final_img)
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}_base.tiff"), base_img)

                all_params.append(params)
                pbar.update(1)

        np.save(os.path.join(output_dir, "all_parameters.npy"), all_params, allow_pickle=True)
        return all_params

    def reproduce_from_params(self, params, override_kernel_size=None, override_sigma=None,
                              override_snr=None, override_z_scale=None, override_noise_seed=None):
        """Complete reproduction"""
        # Restore basic settings
        if isinstance(params['img_size'], str):
            self.img_size = tuple(map(int, params['img_size'].strip('()').split(',')))
        else:
            self.img_size = tuple(params['img_size'])

        # Restore initial Z slice count (before sampling)
        self.z_slices = params.get('initial_z_slices', params.get('z_slices', 5))
        self.is_3d = self.z_slices > 1

        # Override parameters
        kernel_size = override_kernel_size if override_kernel_size is not None else params.get('thicken_kernel_size')
        sigma = override_sigma if override_sigma is not None else params.get('sigma')
        snr = override_snr if override_snr is not None else params.get('noise_snr')
        z_scale = override_z_scale if override_z_scale is not None else params.get('z_scale', 1.0)
        noise_seed = override_noise_seed if override_noise_seed is not None else params.get('noise_random_seed')

        # Rebuild structure
        if self.is_3d:
            base_img = np.zeros((self.z_slices, *self.img_size), dtype=np.uint8)
        else:
            base_img = np.zeros(self.img_size, dtype=np.uint8)

        filaments_data = params['filaments']

        # Improved reproduction logic: use linear interpolation to fill
        for filament in filaments_data:
            points = np.array(filament['points'])
            z_coords = filament.get('z_coords', [])
            z_layer = filament.get('z_layer', 0)

            # 3D drawing logic
            if self.is_3d and z_coords:
                for j in range(len(points) - 1):
                    p1 = points[j]
                    z1 = z_coords[min(j, len(z_coords) - 1)]
                    p2 = points[j + 1]
                    z2 = z_coords[min(j + 1, len(z_coords) - 1)]

                    dist = max(abs(p2 - p1), abs(p2 - p1), abs(z2 - z1))
                    steps = int(np.ceil(dist * 1.5))

                    for k in range(steps + 1):
                        t = k / steps if steps > 0 else 0
                        curr_x = int(p1 + (p2 - p1) * t)
                        curr_y = int(p1 + (p2 - p1) * t)
                        curr_z = int(z1 + (z2 - z1) * t)

                        if (0 <= curr_x < self.img_size and
                                0 <= curr_y < self.img_size and
                                0 <= curr_z < self.z_slices):
                            base_img[curr_z, curr_y, curr_x] = 255

            elif self.is_3d:  # Compatible with old parameters
                z = int(z_layer)
                for point in points:
                    if (0 <= point < self.img_size and
                            0 <= point < self.img_size and
                            0 <= z < self.z_slices):
                        base_img[z, point, point] = 255
            else:  # 2D
                valid_mask = (points[:, 0] >= 0) & (points[:, 0] < self.img_size) & \
                             (points[:, 1] >= 0) & (points[:, 1] < self.img_size)
                pts = points[valid_mask]
                base_img[pts[:, 1], pts[:, 0]] = 255

        # Physical pipeline
        thick_img = self.thicken_filaments(base_img, kernel_size)
        blurred_img = self.add_blur(thick_img, sigma)
        noisy_img = self.add_noise(blurred_img, snr, noise_seed)

        if self.is_3d and z_scale < 1.0:
            final_img = self.apply_anisotropic_sampling(noisy_img, z_scale)
            base_img_final = self.apply_anisotropic_sampling(base_img, z_scale)
        else:
            final_img = noisy_img
            base_img_final = base_img

        return final_img, base_img_final

    def batch_reproduce_from_npy(self, param_file_path, output_dir, **kwargs):
        """Batch reproduce images from NPY file"""
        all_params = np.load(param_file_path, allow_pickle=True)
        os.makedirs(output_dir, exist_ok=True)

        with tqdm(total=len(all_params), desc="Reproducing images") as pbar:
            for params in all_params:
                img_id = params.get('image_id', 'unknown')

                # Reproduce image
                final_img, base_img = self.reproduce_from_params(params, **kwargs)

                # Save
                if self.is_3d:
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}.tiff"), final_img, imagej=True)
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}_base.tiff"), base_img, imagej=True)
                else:
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}.tiff"), final_img)
                    tifffile.imwrite(os.path.join(output_dir, f"{img_id}_base.tiff"), base_img)

                pbar.update(1)


# Example usage
if __name__ == "__main__":
    # 1. Generate synthetic images (2D)
    print("Generating 2D synthetic images...")
    gen_2d = SyntheticFilamentGenerator(img_size=(512, 512), z_slices=0)
    gen_2d.batch_generate(
        output_dir=,
        num_images=20,
        num_filaments=20,
        crossing_prob=0.4,
        curvature_range=(0.1, 0.2),
        cross_angle_range=(45, 90),
        kernel_size_range=(1, 2),
        sigma_range=1,
        snr_range=10,
        cross_length_range=(20, 80),
        length_range=(40, 120)
    )

    # 2. Batch reproduce images (2D) - must use batch_reproduce_from_npy
    print("\nBatch reproducing images...")
    npy_path =
    output_dir =
    if os.path.exists(npy_path):
        gen_2d.batch_reproduce_from_npy(npy_path, output_dir,
                                        override_kernel_size=1, override_sigma=1,
                                        override_snr=10)

    # 3. Generate 3D images (Standard benchmark set)
    # Modification suggestion: Unify num_filaments to 20 as the Anchor Point for all tests
    print("\nGenerating 3D synthetic stack (Base/Standard)...")
    gen_3d = SyntheticFilamentGenerator(img_size=(512, 512), z_slices=64)
    gen_3d.batch_generate(
        output_dir=",
        num_images=20,
        num_filaments=20,
        crossing_prob=0.4,
        curvature_range=(0.1, 0.2),
        cross_angle_range=(45, 90),
        kernel_size_range=1,
        sigma_range=1,
        snr_range=10,
        cross_length_range=(20, 80),
        length_range=(40, 120),
        z_variation=6,
        z_scale=1.0
    )

    npy_path =
    base_output_dir =

    if os.path.exists(npy_path):
        # 4. Anisotropy stress test
        print("\nAnisotropy stress test...")
        test_scenarios = [
            {"z_scale": 1, "desc": "Isotropic"},
            {"z_scale": 0.9, "desc": "Isotropic"},
            {"z_scale": 0.8, "desc": "Mild Anisotropy"},
            {"z_scale": 0.7, "desc": "Mild Anisotropy"},
            {"z_scale": 0.6, "desc": "Mild Anisotropy"},
            {"z_scale": 0.5, "desc": "Mild Anisotropy"},
            {"z_scale": 0.4, "desc": "Moderate Anisotropy"},
            {"z_scale": 0.3, "desc": "Moderate Anisotropy"},
            {"z_scale": 0.2, "desc": "High Anisotropy"},
        ]
        for scenario in test_scenarios:
            print(f"Reproducing {scenario['desc']} (z_scale={scenario['z_scale']})...")
            output_dir = os.path.join(base_output_dir, f"zscale_{scenario['z_scale']}")
            # Fix: Use batch_reproduce_from_npy
            gen_3d.batch_reproduce_from_npy(npy_path, output_dir,
                                            override_z_scale=scenario['z_scale'],
                                            override_kernel_size=1, override_sigma=1, override_snr=10)

        # 5. Blur stress test
        print("\nBlur stress test...")
        blur_scenarios = [
            {"sigma": 1.0, "desc": "Mild blur"},
            {"sigma": 1.25, "desc": "Moderate blur"},
            {"sigma": 1.5, "desc": "Moderate blur"},
            {"sigma": 1.75, "desc": "Moderate blur"},
            {"sigma": 2.0, "desc": "Moderate blur"},
            {"sigma": 2.25, "desc": "heavy blur"},
            {"sigma": 2.5, "desc": "heavy blur"},
            {"sigma": 2.75, "desc": "heavy blur"},
            {"sigma": 3.0, "desc": "heavy blur"}
        ]
        for scenario in blur_scenarios:
            print(f"Reproducing blur: {scenario['desc']} (sigma={scenario['sigma']})...")
            output_dir = os.path.join(base_output_dir, f"blur_{scenario['sigma']}")
            # Fix: Use batch_reproduce_from_npy
            gen_3d.batch_reproduce_from_npy(npy_path, output_dir,
                                            override_sigma=scenario['sigma'],
                                            override_z_scale=1, override_kernel_size=1, override_snr=10)

        # 6. Noise stress test
        print("\nNoise stress test...")
        noise_scenarios = [
            {"snr": 10, "desc": "Mild noise"},
            {"snr": 9, "desc": "Mild noise"},
            {"snr": 8, "desc": "Mild noise"},
            {"snr": 7, "desc": "Moderate noise"},
            {"snr": 6, "desc": "Moderate noise"},
            {"snr": 5, "desc": "Moderate noise"},
            {"snr": 4, "desc": "Moderate noise"},
            {"snr": 3, "desc": "Heavy noise"},
            {"snr": 2, "desc": "Heavy noise"}
        ]
        for scenario in noise_scenarios:
            print(f"Reproducing noise: {scenario['desc']} (snr={scenario['snr']})...")
            output_dir = os.path.join(base_output_dir, f"noise_{scenario['snr']}")
            # Fix: Use batch_reproduce_from_npy
            gen_3d.batch_reproduce_from_npy(npy_path, output_dir,
                                            override_snr=scenario['snr'],
                                            override_z_scale=1, override_kernel_size=1, override_sigma=1)

    # 7. Structural complexity stress test
    print("\nStructural complexity stress test...")
    # Modification suggestion: Range starts from 20 to align with baseline
    num_filaments_range = []

    fixed_params = {
        'curvature_range': (0.1, 0.2),
        'cross_angle_range': (45, 90),
        'kernel_size_range': 1,
        'sigma_range': 1,
        'snr_range': 10,
        'cross_length_range': (20, 80),
        'length_range': (40, 120),
        'z_variation': 6,
        'z_scale': 1.0
    }

    for num_filaments in num_filaments_range:
        print(f"Generating structural complexity test: num_filaments={num_filaments}")
        output_dir = os.path.join(base_output_dir, f"num_{num_filaments}")
        # Note: This must be generate (generate new structure), not reproduce
        gen_3d.batch_generate(
            output_dir=output_dir,
            num_images=20,
            num_filaments=num_filaments,
            crossing_prob=0.4,
            **fixed_params
        )

    print("\nAll test generations complete!")