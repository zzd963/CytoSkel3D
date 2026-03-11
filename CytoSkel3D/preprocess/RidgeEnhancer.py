import os
import numpy as np
from skimage import io, filters, morphology, exposure, feature, util, img_as_ubyte, img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.transform import rotate


class RidgeEnhancer:
    """Ridge enhancement processor (supports 2D/3D images)"""

    def __init__(self, params: dict, im_info):
        self.im_info = im_info
        self.params = params
        self.is_3d = im_info.get_memmap('processed').ndim == 3

        # Tubular structure size parameters
        self.min_radius_px = params.get('min_radius_px', 2)
        self.max_radius_px = params.get('max_radius_px', 10)

        # Ridge enhancement parameters
        self.method = params.get('ridge_method', 'Sato')
        self.method_params = params.get('ridge_params', {}).get(self.method, {})

        # Whether to use radius to calculate sigma
        self.use_radius_for_sigma = params.get('use_radius_for_sigma', False)

        # Maximum number of iterations
        self.max_iter = params.get('max_iter', 3)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Execute ridge enhancement processing"""
        processors = {
            'Sato': self._sato,
            'Frangi': self._frangi,
            'Frangi_iteration': self._frangi_iter,
            'Frangi_Sato': self._frangi_sato,
            'Meijering': self._meijering,
            'Hessian': self._hessian,
            'multi_orient_skel': self._multi_orient_skel
        }

        if self.method not in processors:
            raise ValueError(f"Unsupported ridge enhancement method: {self.method}")

        # Create foreground mask
        foreground_mask = image > 0

        # Execute ridge enhancement
        enhanced = processors[self.method](image)

        # Apply foreground mask: only keep enhanced results in areas where original image > 0
        enhanced = np.where(foreground_mask, enhanced, 0)

        return enhanced

    def _get_sigmas(self):
        """Get sigma values based on configuration"""
        if self.use_radius_for_sigma:
            # Use radius to calculate sigma range
            return np.linspace(  # The radius of the tubular structure is approximately sqrt(2) times sigma
                self.min_radius_px / 2,
                self.max_radius_px / 2,  # Modified to /2 to get better coverage
                num=7  # Generate 7 sigma values by default
            )
        else:
            # Use sigma settings in ridge_params
            return self.method_params.get('sigmas', np.linspace(1, 4, 7))

    def _sato(self, image: np.ndarray) -> np.ndarray:
        """Sato filter implementation"""
        # Get sigma values
        sigmas = self._get_sigmas()

        params = {
            'sigmas': sigmas,
            'black_ridges': False
        }
        params.update({k: v for k, v in self.method_params.items() if k != 'sigmas'})

        return filters.sato(image, **params)

    def _frangi(self, image: np.ndarray) -> np.ndarray:
        """Frangi filter implementation"""
        # Get sigma values
        sigmas = self._get_sigmas()

        params = {
            'sigmas': sigmas,
            'black_ridges': False
        }
        params.update({k: v for k, v in self.method_params.items() if k != 'sigmas'})

        return filters.frangi(image, **params)

    def _frangi_iter(self, image: np.ndarray) -> np.ndarray:
        """Iterative Frangi filter implementation, until results stabilize"""
        # Get initial parameters
        sigmas = self._get_sigmas()
        params = {
            'sigmas': sigmas,
            'black_ridges': False
        }
        params.update({k: v for k, v in self.method_params.items() if k != 'sigmas'})

        # Iterative enhancement parameters
        # max_iter = 2  # Maximum number of iterations
        tolerance = 1e-4  # Convergence threshold
        prev_enhanced = image.copy()  # Initialize previous result

        for i in range(self.max_iter):
            # Apply Frangi filter
            enhanced = filters.frangi(prev_enhanced, **params)

            # Calculate difference between current and previous results
            diff = np.mean(np.abs(enhanced - prev_enhanced))

            # Check convergence condition (difference is less than threshold or max iterations reached)
            if diff < tolerance:
                print(f"Frangi converged after {i + 1} iterations")
                return enhanced

            prev_enhanced = enhanced  # Update previous result

        print(f"Frangi reached max iterations ({self.max_iter})")
        return prev_enhanced

    def _frangi_sato(self, image: np.ndarray) -> np.ndarray:
        """Combined Frangi+Sato filter"""
        # Get sigma values
        sigmas = self._get_sigmas()

        params = {
            'sigmas': sigmas,
            'black_ridges': False
        }
        params.update({k: v for k, v in self.method_params.items() if k != 'sigmas'})

        image_tmp = filters.frangi(image, **params)
        return filters.sato(image_tmp, **params)

    def _meijering(self, image: np.ndarray) -> np.ndarray:
        """Meijering filter implementation"""
        # Get sigma values
        sigmas = self._get_sigmas()

        params = {
            'sigmas': sigmas,
            'black_ridges': False
        }
        params.update({k: v for k, v in self.method_params.items() if k != 'sigmas'})

        return filters.meijering(image, **params)

    def _hessian(self, image: np.ndarray) -> np.ndarray:
        """Hessian filter implementation (unified processing for 2D/3D)"""
        sigma = self.method_params.get('sigma', 1.0)

        if self.is_3d:
            # Process 3D image layer by layer
            enhanced = np.zeros_like(image)
            for z in range(image.shape):
                layer = image[z]
                H = hessian_matrix(layer, sigma=sigma)
                _, lambda2 = hessian_matrix_eigvals(H)
                enhanced[z] = exposure.rescale_intensity(lambda2, out_range=(0, 1))
            return util.invert(enhanced)
        else:
            # Process 2D image
            H = hessian_matrix(image, sigma=sigma)
            _, lambda2 = hessian_matrix_eigvals(H)
            enhanced = exposure.rescale_intensity(lambda2, out_range=(0, 1))
            return util.invert(enhanced)


    def _multi_orient_skel(self, image: np.ndarray) -> np.ndarray:
        """Improved multi-orientation skeleton generation method"""
        # Get parameters, defaulting to 6 orientations (0°, 30°, 60°, 90°, 120°, 150°)
        angles = self.method_params.get('angles',)

        # Get sigma values
        sigmas = self._get_sigmas()

        # Initialize result array
        if self.is_3d:
            ridges = np.zeros_like(image, dtype=np.float32)
        else:
            ridges = np.zeros_like(image, dtype=np.float32)

        # Apply Hessian filtering for each angle
        for angle in angles:
            # For 3D images, process layer by layer
            if self.is_3d:
                angle_response = np.zeros_like(image, dtype=np.float32)

                for z in range(image.shape):
                    # Get current layer
                    layer = image[z]

                    # Rotate image (using bilinear interpolation)
                    rotated = rotate(layer, angle, resize=False, mode='reflect', order=1)

                    # Initialize angle response for current layer
                    layer_response = np.zeros_like(rotated)

                    # Multi-scale Hessian filtering
                    for sigma in sigmas:
                        # Calculate Hessian matrix and eigenvalues
                        H = hessian_matrix(rotated, sigma=sigma, mode='reflect')
                        _, lambda2 = hessian_matrix_eigvals(H)

                        # Use the maximum lambda2 value (absolute value)
                        layer_response = np.maximum(layer_response, np.abs(lambda2))

                    # Rotate back to original orientation
                    layer_response = rotate(layer_response, -angle, resize=False, mode='reflect', order=1)

                    # Crop to original size
                    if layer_response.shape != layer.shape:
                        # Calculate padding amount
                        dy = layer.shape - layer_response.shape
                        dx = layer.shape - layer_response.shape

                        # Pad only when necessary
                        if dy > 0 or dx > 0:
                            # Calculate top, bottom, left, right padding amounts
                            pad_width = (
                                (max(0, dy // 2), max(0, dy - dy // 2)),
                                (max(0, dx // 2), max(0, dx - dx // 2))
                            )

                            # Pad only when padding is required
                            layer_response = np.pad(
                                layer_response,
                                pad_width[:layer_response.ndim],  # Adapt based on dimensions
                                mode='constant'
                            )

                        # Crop excess parts (if any)
                        layer_response = layer_response[:layer.shape, :layer.shape]

                    # Store response of current layer
                    angle_response[z] = layer_response
            else:
                # Process 2D image
                rotated = rotate(image, angle, resize=False, mode='reflect', order=1)

                # Initialize angle response
                angle_response = np.zeros_like(rotated)

                # Multi-scale Hessian filtering
                for sigma in sigmas:
                    # Calculate Hessian matrix and eigenvalues
                    H = hessian_matrix(rotated, sigma=sigma, mode='reflect')
                    _, lambda2 = hessian_matrix_eigvals(H)

                    # Use the maximum lambda2 value (absolute value)
                    angle_response = np.maximum(angle_response, np.abs(lambda2))

                # Rotate back to original orientation
                angle_response = rotate(angle_response, -angle, resize=False, mode='reflect', order=1)

                # Crop to original size
                if angle_response.shape != image.shape:
                    # Calculate padding amount
                    dy = image.shape - angle_response.shape
                    dx = image.shape - angle_response.shape

                    # Pad only when necessary
                    if dy > 0 or dx > 0:
                        # Calculate top, bottom, left, right padding amounts
                        pad_width = (
                            (max(0, dy // 2), max(0, dy - dy // 2)),
                            (max(0, dx // 2), max(0, dx - dx // 2))
                        )

                        # Pad only when padding is required
                        angle_response = np.pad(
                            angle_response,
                            pad_width[:angle_response.ndim],  # Adapt based on dimensions
                            mode='constant'
                        )

                    # Crop excess parts (if any)
                    angle_response = angle_response[:image.shape, :image.shape]

            # Accumulate responses from all orientations
            ridges += angle_response

        # Normalize
        ridges /= len(angles)

        # Ensure values are within 0-1 range
        ridges = exposure.rescale_intensity(ridges, out_range=(0, 1))

        return ridges.astype(np.float32)