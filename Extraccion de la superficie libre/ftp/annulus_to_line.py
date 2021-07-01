from typing import Tuple
import numpy as np

def get_polar_band(image: np.ndarray,
                   center: Tuple[float, float],
                   radius_limits: Tuple[float, float],
                   band_resolution: int):

    initial_radius, final_radius = radius_limits

    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, band_resolution),
                            np.arange(initial_radius, final_radius))

    xcart = (r * np.cos(theta) + center[0]).astype(int)
    ycart = (r * np.sin(theta) + center[1]).astype(int)

    image_band = image[ycart, xcart].reshape(final_radius-initial_radius, band_resolution)
    return image_band

def find_annulus_region(annulus_mask, band_resolution):

    half_width = annulus_mask.shape[0]//2
    annulus_mask_band = get_polar_band(annulus_mask,
                                       center=(half_width, half_width),
                                       radius_limits=(0, half_width),
                                       band_resolution=band_resolution)

    initial_radius = int(half_width-np.max(annulus_mask_band.sum(0)))

    final_radius = half_width
    annulus_region_mask = annulus_mask_band[initial_radius:final_radius, :].astype(bool)

    return initial_radius, final_radius, annulus_region_mask

def get_polar_band_average(annulus: np.ndarray,
                           center: Tuple[float, float],
                           radius_limits: Tuple[float, float],
                           annulus_region_mask: np.ndarray,
                           band_resolution: int):

    annulus_band = get_polar_band(annulus, center=center,
                                  radius_limits=radius_limits,
                                  band_resolution=band_resolution)
    band_average = np.nanmean(np.where(annulus_region_mask, annulus_band, np.nan), 0)

    return band_average

def get_spatiotemporal_diagram(height_fields_path: str,
                               band_resolution: int=3000):
    """
    Toma un HDF5 con dos grupos: uno con las imagenes ya procesadas
    por ftp y otro con la máscara utilizada.

    Retorna el diagrama espaciotemporal.
    """
    f = h5py.File(height_fields_path, 'r')
    annulus_array = f['height_fields/annulus']
    annulus_mask = f['masks/annulus'][:, :, 0]

    *radius_limits, annulus_region_mask = find_annulus_region(annulus_mask,
                                                              band_resolution)

    half_width = annulus_array.shape[0]//2
    center = [half_width]*2

    n_images = annulus_array.shape[-1]
    n_chunks = annulus_array.chunks[-1]*2

    spatiotemporal_diagram = np.zeros(shape=(n_images, band_resolution))
    for i in range(n_images//n_chunks): # TODO: Pedir el annulus_array[:, :, i] de a chunks
        print(i)

        annulus_chunk = annulus_array[:, :, i*n_chunks: (i+1)*n_chunks]
        for j in range(n_chunks):
            annulus_band_average = get_polar_band_average(
                                           annulus_chunk[:, :, j], center=center,
                                           radius_limits=radius_limits,
                                           annulus_region_mask=annulus_region_mask,
                                           band_resolution=band_resolution)

            spatiotemporal_diagram[i*n_chunks+j, :] = annulus_band_average

    return spatiotemporal_diagram

if __name__ == '__main__':
    import h5py
    import matplotlib
    from matplotlib import pyplot as plt
    import cProfile
    import pstats

    #  height_fields_path = '/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5'
    height_fields_path = '/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2019-10-09-a200-f20-fmod0-annulus-PRO.hdf5'
    #  annulus_mask = np.asarray(f['masks/annulus'][:, :, 0])
    #  with cProfile.Profile() as pr:
        #  get_spatiotemporal_diagram(height_fields_path)

    #  stats = pstats.Stats(pr)
    #  stats.sort_stats(pstats.SortKey.TIME)
    #  stats.print_stats()

    spatiotemporal_diagram = get_spatiotemporal_diagram(height_fields_path)
    plt.imshow(spatiotemporal_diagram)
    plt.show()

    #  f = h5py.File('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5', 'r')
    #  f = h5py.File('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2019-10-09-a200-f20-fmod0-annulus-PRO.hdf5', 'r')



    #  annulus = f['height_fields/annulus']
    #  annulus_mask = f['masks/annulus']

    #  annulus = annulus[:, :, 40]
    #  annulus_mask = annulus_mask[:, :, 0]

    #  *radius_limits, annulus_region_mask = find_annulus_region(annulus_mask)

    #  half_width = annulus.shape[0]//2
    #  center = [half_width]*2
    #  annulus_band_average = get_polar_band_average(
                                   #  annulus, center=center,
                                   #  radius_limits=radius_limits,
                                   #  annulus_region_mask=annulus_region_mask
                                   #  )

    #  plt.plot(annulus_band_average)
    #  #  plt.imshow(img, cmap='gray')
    #  plt.show()

