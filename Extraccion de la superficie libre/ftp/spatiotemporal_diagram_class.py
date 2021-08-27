# WIP

import h5py
import numpy as np

class SpatioTemporalDiagram():
    def __init__(self, hdf5_folder):

        self.hdf5_folder = hdf5_folder

        f = h5py.File(hdf5_folder+'FTP.hdf5')

        self.annulus_mask = f['masks/annulus']
        self.annulus_height_fields = f['height_fields/annulus']

    def _get_polar_strip(image: np.ndarray,
                         center: Tuple[float, float],
                         radius_limits: Tuple[float, float],
                         strip_resolution: int):

        initial_radius, final_radius = radius_limits

        theta, r = np.meshgrid(np.linspace(0, 2*np.pi, strip_resolution),
                                np.arange(initial_radius, final_radius))

        xcart = (r * np.cos(theta) + center[0]).astype(int)
        ycart = (r * np.sin(theta) + center[1]).astype(int)

        image_strip = image[ycart, xcart].reshape(final_radius-initial_radius, strip_resolution)
        return image_strip
