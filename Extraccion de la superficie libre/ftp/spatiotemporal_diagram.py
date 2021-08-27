from typing import Tuple
import logging

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt = '%H:%M:%S')

def get_polar_strip(image: np.ndarray,
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

def find_annulus_region(annulus_mask, center, annulus_radii, strip_resolution):

    half_width = annulus_mask.shape[0]//2
    annulus_mask_strip = get_polar_strip(annulus_mask,
                                         center=center,
                                         radius_limits=(0, half_width),
                                         strip_resolution=strip_resolution)

    #  initial_radius = int(half_width-np.max(annulus_mask_strip.sum(0)))
    #  final_radius = half_width

    initial_radius, final_radius = annulus_radii

    annulus_region_mask = annulus_mask_strip[initial_radius:final_radius, :].astype(bool)

    return initial_radius, final_radius, annulus_region_mask

def get_polar_strip_average(annulus: np.ndarray,
                           center: Tuple[float, float],
                           radius_limits: Tuple[float, float],
                           annulus_region_mask: np.ndarray,
                           strip_resolution: int):

    annulus_strip = get_polar_strip(annulus, center=center,
                                  radius_limits=radius_limits,
                                  strip_resolution=strip_resolution)
    strip_average = np.nanmean(np.where(annulus_region_mask, annulus_strip, np.nan), 0)

    return strip_average

def get_st_diagram(ftp_hdf5_path: str,
                               strip_resolution: int=3000):
    """
    Toma un HDF5 con dos grupos: uno con las imagenes ya procesadas
    por ftp y otro con la máscara utilizada.

    Retorna el diagrama espaciotemporal.
    """

    f = h5py.File(ftp_hdf5_path, 'r')
    annulus_array = f['height_fields/annulus']
    annulus_mask = f['masks/annulus'][:, :]
    annulus_mask_info = f['masks/annulus'].attrs

    center = annulus_mask_info['center']
    annulus_radii = annulus_mask_info['annulus_radii']

    annulus_region_mask = get_polar_strip(annulus_mask, center, annulus_radii,
                                          strip_resolution=3000)

    n_images = annulus_array.shape[-1]
    img_per_chunk = annulus_array.chunks[-1]

    n_chunks = np.ceil(n_images/img_per_chunk).astype(int)

    st_diagram = np.zeros(shape=(n_images, strip_resolution))

    logging.info('ST: Inicio del cálculo del diagrama espacio-temporal')

    for i in range(n_chunks-1):

        annulus_chunk = annulus_array[:, :, i*img_per_chunk: (i+1)*img_per_chunk]
        for j in range(img_per_chunk):
            annulus_strip_average = get_polar_strip_average(
                                           annulus_chunk[:, :, j], center=center,
                                           radius_limits=annulus_radii,
                                           annulus_region_mask=annulus_region_mask,
                                           strip_resolution=strip_resolution)

            #  annulus_strip_average -= annulus_strip_average.mean()
            st_diagram[i*img_per_chunk+j, :] = annulus_strip_average

        logging.info(f'ST: {i+1}/{n_chunks} chunks calculados')
    logging.info('ST: END')

    # Guardo el último chunk aparte porque podría ser más corto
    annulus_chunk = annulus_array[:, :, (n_chunks-1)*img_per_chunk:n_images]
    for j in range(annulus_chunk.shape[-1]):
        annulus_strip_average = get_polar_strip_average(
                                       annulus_chunk[:, :, j], center=center,
                                       radius_limits=annulus_radii,
                                       annulus_region_mask=annulus_region_mask,
                                       strip_resolution=strip_resolution)

        #  annulus_strip_average -= annulus_strip_average.mean()
        st_diagram[(n_chunks-1)*img_per_chunk+j, :] = annulus_strip_average

    return st_diagram

def create_st_hdf5(hdf5_folder):
    st_diagram = get_st_diagram(hdf5_folder+'FTP.hdf5')

    f = h5py.File(hdf5_folder+'ST.hdf5', 'w')
    f.create_dataset('spatiotemporal_diagram', data=st_diagram)
    f.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    med_folder = '../../Mediciones_FaradayWaves/MED11 - 0730/'
    hdf5_folder = med_folder+'HDF5/'

    #  create_st_hdf5(hdf5_folder)

    f = h5py.File(hdf5_folder+'ST.hdf5', 'r')
    st_diagram = np.array(f['spatiotemporal_diagram'])
    st_diagram = (st_diagram.T - st_diagram.mean(1)).T

    plt.imshow(st_diagram)
    plt.show()
