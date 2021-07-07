from typing import Tuple
import h5py
import numpy as np

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

def find_annulus_region(annulus_mask, strip_resolution):

    half_width = annulus_mask.shape[0]//2
    annulus_mask_strip = get_polar_strip(annulus_mask,
                                       center=(half_width, half_width),
                                       radius_limits=(0, half_width),
                                       strip_resolution=strip_resolution)

    initial_radius = int(half_width-np.max(annulus_mask_strip.sum(0)))

    final_radius = half_width
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

def get_spatiotemporal_diagram(height_fields_path: str,
                               strip_resolution: int=3000):
    """
    Toma un HDF5 con dos grupos: uno con las imagenes ya procesadas
    por ftp y otro con la máscara utilizada.

    Retorna el diagrama espaciotemporal.
    """
    f = h5py.File(height_fields_path, 'r')
    annulus_array = f['height_fields/annulus']
    annulus_mask = f['masks/annulus'][:, :, 0]

    *radius_limits, annulus_region_mask = find_annulus_region(annulus_mask,
                                                              strip_resolution)

    half_width = annulus_array.shape[0]//2
    center = [half_width]*2

    n_images = annulus_array.shape[-1]
    n_chunks = annulus_array.chunks[-1]*2

    spatiotemporal_diagram = np.zeros(shape=(n_images, strip_resolution))
    for i in range(n_images//n_chunks):
        print(i)

        annulus_chunk = annulus_array[:, :, i*n_chunks: (i+1)*n_chunks]
        for j in range(n_chunks):
            annulus_strip_average = get_polar_strip_average(
                                           annulus_chunk[:, :, j], center=center,
                                           radius_limits=radius_limits,
                                           annulus_region_mask=annulus_region_mask,
                                           strip_resolution=strip_resolution)

            spatiotemporal_diagram[i*n_chunks+j, :] = annulus_strip_average

    return spatiotemporal_diagram

if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt
    import cProfile
    import pstats

    #  height_fields_path = '/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5'
    height_fields_path = '/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2019-10-09-a200-f20-fmod0-annulus-PRO.hdf5'
    #  spatiotemporal_diagram = get_spatiotemporal_diagram(height_fields_path)

    #  plt.imshow(spatiotemporal_diagram, cmap='gray')
    #  plt.colorbar()
    #  plt.clim(-10, 10)

    #  plt.imshow(spatiotemporal_diagram, cmap='gray')
    #  plt.colorbar()

    #  plt.show()
    #  plt.plot(spatiotemporal_diagram[1950, :]-spatiotemporal_diagram[1950, :].mean())
    #  plt.plot(spatiotemporal_diagram[2499, :]-spatiotemporal_diagram[2499, :].mean())
    #  plt.show()

    #  with cProfile.Profile() as pr:
        #  get_spatiotemporal_diagram(height_fields_path)

    #  stats = pstats.Stats(pr)
    #  stats.sort_stats(pstats.SortKey.TIME)
    #  stats.print_stats()

    height_fields_path ='/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5'
    #  height_fields_path = '/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2019-10-09-a200-f20-fmod0-annulus-PRO.hdf5'
    f = h5py.File(height_fields_path)


    annulus = f['height_fields/annulus']
    annulus_mask = f['masks/annulus']
    annulus = annulus[:, :, 40]
    annulus_mask = annulus_mask[:, :, 0]
    strip_resolution = 3000

    #  half_width = annulus.shape[0]//2
    #  center = [half_width]*2

    #  *radius_limits, annulus_region_mask = find_annulus_region(annulus_mask, strip_resolution)

    #  annulus_strip_average = get_polar_strip_average(
                                   #  annulus, center=center,
                                   #  radius_limits=radius_limits,
                                   #  annulus_region_mask=annulus_region_mask,
                                   #  strip_resolution=strip_resolution
                                   #  )

    #  plt.plot(annulus_strip_average)

    half_width = annulus_mask.shape[0]//2
    center = [half_width]*2
    annulus_mask_strip = get_polar_strip(annulus_mask,
                                       center=(half_width, half_width),
                                       radius_limits=(0, half_width),
                                       strip_resolution=strip_resolution)

    plt.imshow(annulus_mask_strip, cmap='gray')
    plt.show()
