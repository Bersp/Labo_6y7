import logging
from typing import Tuple

import h5py
import numpy as np
from unwrap import unwrap

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')


def get_polar_strip(image: np.ndarray, center: Tuple[int, int],
                    radius_limits: Tuple[int, int], strip_resolution: int):
    """
    Funcion para pasar de anillo en cartesianas a banda en coordenadas polares.
    Esta banda tiene un ancho igual a la diferencia entre dos radios límites dados.

    Parameters
    ----------
    image : 2-D array
        La imagen a procesar.
    center : tuple of ints.
        Centro del anillo (vertical, horizonatal). e.g. (512, 512).
    radius_limits : tuple of ints.
        Radio menor y radio mayor del anillo. e.g., (420, 470).
    strip_resolution : int.
        Cantidad de puntos de la primera dimensión de la banda (strip).


    Returns
    -------
    image_strip : 2-D array
        Banda de dimensiones (strip_resolution, diff(radius_limits)).


    Scheme
    ------
           0 
          o  o          
       o        o                           
      o          o 3              o o o o o o o o o o o o
    1 o          o     ---->     0     1     2     3
       o        o   
          o  o     
            2                          
    """

    initial_radius, final_radius = radius_limits

    theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, strip_resolution),
                           np.arange(initial_radius, final_radius))

    theta = theta - np.pi / 2
    xcart = -(r * np.cos(theta) + center[0]).astype(int)
    ycart = (r * np.sin(theta) + center[1]).astype(int)

    image_strip = image[ycart, xcart].reshape(final_radius - initial_radius,
                                              strip_resolution)
    return image_strip


def find_annulus_region(annulus_mask, center, annulus_radii, strip_resolution):
    """
    TODO: Docstring.

    Parameters
    ----------
    annulus_mask :  TODO

    Returns
    -------
    TODO

    """

    half_width = annulus_mask.shape[0] // 2
    annulus_mask_strip = get_polar_strip(annulus_mask,
                                         center=center,
                                         radius_limits=(0, half_width),
                                         strip_resolution=strip_resolution)

    initial_radius, final_radius = annulus_radii

    annulus_region_mask = annulus_mask_strip[
        initial_radius:final_radius, :].astype(bool)

    return initial_radius, final_radius, annulus_region_mask


def get_polar_strip_average(annulus: np.ndarray, center: Tuple[float, float],
                            radius_limits: Tuple[float, float],
                            annulus_region_mask: np.ndarray,
                            strip_resolution: int):

    annulus_strip = get_polar_strip(annulus,
                                    center=center,
                                    radius_limits=radius_limits,
                                    strip_resolution=strip_resolution)
    masked_annulus_strip = np.where(annulus_region_mask, annulus_strip, np.nan)

    strip_average = np.nanmean(masked_annulus_strip, 0)
    strip_std = np.nanstd(masked_annulus_strip, 0)

    return strip_average, strip_std


def vertical_unwraping(st_diagram):
    """
    TODO: Doctring for vertical_unwraping
    TODO: Hacer un promedio de muchas lineas verticales.
    Actualmente solo toma una (la 100).
    """
    gap = st_diagram[:, 100] - np.unwrap(st_diagram[:, 100])

    gap = np.expand_dims(gap, 1)
    st_diagram = st_diagram - gap

    st_diagram = unwrap(st_diagram)

    st_diagram -= np.expand_dims(st_diagram.mean(1), 1)

    return st_diagram


def get_st_diagram(ftp_hdf5_path: str,
                   transform_to_mm=False,
                   strip_resolution: int = 3000):
    """
    TODO: Doctring for get_st_diagram

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

    annulus_region_mask = get_polar_strip(annulus_mask,
                                          center,
                                          annulus_radii,
                                          strip_resolution=3000)

    n_images = annulus_array.shape[-1]
    img_per_chunk = annulus_array.chunks[-1]

    n_chunks = np.ceil(n_images / img_per_chunk).astype(int)

    st_diagram = np.zeros(shape=(n_images, strip_resolution))
    st_error = np.zeros(shape=(n_images, strip_resolution))

    logging.info('ST: Inicio del cálculo del diagrama espacio-temporal')

    for i in range(n_chunks - 1):

        annulus_chunk = annulus_array[:, :, i * img_per_chunk:(i + 1) *
                                      img_per_chunk]
        for j in range(img_per_chunk):
            annulus_strip_average, annulus_strip_std = get_polar_strip_average(
                annulus_chunk[:, :, j],
                center=center,
                radius_limits=annulus_radii,
                annulus_region_mask=annulus_region_mask,
                strip_resolution=strip_resolution)

            #  annulus_strip_average -= annulus_strip_average.mean()
            st_diagram[i * img_per_chunk + j, :] = annulus_strip_average
            st_error[i * img_per_chunk + j, :] = annulus_strip_std

        logging.info(f'ST: {i+1}/{n_chunks} chunks calculados')

    # Guardo el último chunk aparte porque podría ser más corto
    annulus_chunk = annulus_array[:, :,
                                  (n_chunks - 1) * img_per_chunk:n_images]
    for j in range(annulus_chunk.shape[-1]):
        annulus_strip_average, annulus_strip_std = get_polar_strip_average(
            annulus_chunk[:, :, j],
            center=center,
            radius_limits=annulus_radii,
            annulus_region_mask=annulus_region_mask,
            strip_resolution=strip_resolution)

        #  annulus_strip_average -= annulus_strip_average.mean()
        st_diagram[(n_chunks - 1) * img_per_chunk +
                   j, :] = annulus_strip_average
        st_error[(n_chunks - 1) * img_per_chunk + j, :] = annulus_strip_std

    # Unwraping vertical
    logging.info(f'ST: Calculando unwraping vertical')
    st_diagram = st_diagram[:, 1000:]
    st_diagram = vertical_unwraping(st_diagram)

    if transform_to_mm:
        L, d, p = f.attrs['L'], f.attrs['d'], f.attrs['p']
        st_diagram = phase_to_height(st_diagram, L, d, p)
        st_error = phase_to_height(st_error, L, d, p)
        print(f'{L = }')
        print(f'{d = }')
        print(f'{p = }')

    logging.info('ST: END')
    return st_diagram, st_error


def phase_to_height(st_diagram, L, d, p):
    """
    TODO: Docstring for phase_to_height.

    Parameters
    ----------
    st_diagram : Diagrama espacio-temporal
    L : Distancia entre el plano de referencia y la cámara
    d : Distancia entre el proyector y la cámara
    p : Longitud de onda del patrón proyectado en mm

    Returns
    -------
    El diagrama en unidades de altura (mm)
    """

    dphase = st_diagram
    return -L * dphase / (2 * np.pi * d / p - dphase)


def create_st_hdf5(hdf5_folder):
    """
    TODO: Doctring for create_st_hdf5
    """
    st_diagram, st_error = get_st_diagram(hdf5_folder + 'FTP.hdf5')

    f = h5py.File(hdf5_folder + 'ST.hdf5', 'w')
    f.create_dataset('spatiotemporal_diagram', data=st_diagram)
    f.create_dataset('spatiotemporal_diagram_error', data=st_error)
    f.close()


def main():
    import matplotlib.pyplot as plt
    import matplotlib
    cmap = matplotlib.cm.viridis
    med_folder = '../../Mediciones/MED63 - Bajada en voltaje - NOTE - 1104/'
    hdf5_folder = med_folder + 'HDF5/'

    # create_st_hdf5(hdf5_folder)

    # f = h5py.File(hdf5_folder + 'ST.hdf5', 'r')
    # st_diagram = np.array(f['spatiotemporal_diagram'])
    # st_error = np.array(f['spatiotemporal_diagram_error'])

    # st_diagram -= np.mean(st_diagram, 0)

    # f = h5py.File(hdf5_folder+'FTP.hdf5', 'r')
    # L, d, p = f.attrs['L'], f.attrs['d'], f.attrs['p']
    # st_diagram = phase_to_height(st_diagram, L, d, p)
    # st_error = phase_to_height(st_error, L, d, p)

    # plt.imshow(st_diagram)
    # plt.colorbar()
    # plt.plot(st_diagram[2200])
    # plt.show()


def delete():
    import matplotlib.pyplot as plt
    import h5py

    hdf5_folder = '/home/bersp/Documents/Labo_6y7/Mediciones/MED63 - Bajada en voltaje - NOTE - 1104/HDF5/'
    # st_diagram, st_error = get_st_diagram(hdf5_folder + 'FTP.hdf5')
    # create_st_hdf5(hdf5_folder)

    f = h5py.File(hdf5_folder + 'FTP.hdf5', 'r')
    f2 = h5py.File(hdf5_folder + 'RAW.hdf5', 'r')

    # image = f2['deformed'][:, :, 0]
    # image = f['height_fields']['annulus'][:,:,0]
    fig, axes = plt.subplots(2, figsize=(12, 8), sharex=True, sharey=False)
    for ax, image in zip(
            axes,
        [f2['reference'][:, :], f['height_fields']['annulus'][:, :, 0]]):
        annulus_mask_info = f['masks/annulus'].attrs
        center = annulus_mask_info['center']
        annulus_radii = annulus_mask_info['annulus_radii']

        annulus_region_mask = get_polar_strip(f['masks/annulus'][:, :],
                                              center,
                                              annulus_radii,
                                              strip_resolution=3000)

        line = get_polar_strip_average(annulus=image,
                                       center=center,
                                       radius_limits=annulus_radii,
                                       annulus_region_mask=annulus_region_mask,
                                       strip_resolution=3000)[0]

        annulus_strip = get_polar_strip(image,
                                        center=center,
                                        radius_limits=annulus_radii,
                                        strip_resolution=3000)
        ax.plot(line)
        # plt.imshow(image)
        # ax.imshow(annulus_strip)
        # ax.colorbar()
    plt.show()


def delete2():
    import matplotlib.pyplot as plt
    import numpy as np

    xx, yy = np.meshgrid(np.arange(1024), np.arange(1024))
    zz = np.arctan2(xx - 512, yy - 512)
    zz = np.cos(2 * np.pi / 20 * xx)

    center = (512, 512)
    rr = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    radii = (480, 510)

    image = zz
    image_mask = np.ones_like(image)
    image_mask[radii[1] < rr] = np.nan
    image_mask[rr < radii[0]] = np.nan

    # strip = get_polar_strip(image*image_mask, center, radii, strip_resolution=3000)

    annulus_region_mask = get_polar_strip(image_mask,
                                          center,
                                          radii,
                                          strip_resolution=3000)

    line = get_polar_strip_average(annulus=image,
                                   center=center,
                                   radius_limits=radii,
                                   annulus_region_mask=annulus_region_mask,
                                   strip_resolution=3000)[0]

    strip = get_polar_strip(image, center, radii, strip_resolution=3000)

    plt.imshow(image)
    plt.figure()
    plt.plot(line)
    plt.figure()
    plt.imshow(strip, aspect='auto')
    plt.colorbar()

    plt.show()

def delete3():
    import matplotlib.pyplot as plt

    hdf5_folder = '/home/bersp/Documents/Labo_6y7/Mediciones/MED63 - Bajada en voltaje - NOTE - 1104/HDF5/'

    f_ftp = h5py.File(hdf5_folder + 'FTP.hdf5', 'r')
    f_raw = h5py.File(hdf5_folder + 'RAW.hdf5', 'r')

    annulus_mask_info = f_ftp['masks/annulus'].attrs
    center = annulus_mask_info['center']
    annulus_radii = annulus_mask_info['annulus_radii']

    image_mask = f_ftp['masks/annulus'][:,:]

    annulus_region_mask = get_polar_strip(image_mask,
                                          center,
                                          annulus_radii,
                                          strip_resolution=3000)

    # line = get_polar_strip_average(annulus=image,
                                   # center=center,
                                   # radius_limits=annulus_radii,
                                   # annulus_region_mask=annulus_region_mask,
                                   # strip_resolution=3000)[0]

    # annulus_strip = get_polar_strip(image,
                                    # center=center,
                                    # radius_limits=annulus_radii,
                                    # strip_resolution=3000)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(annulus_region_mask)
    plt.show()

if __name__ == '__main__':
    # main()
    # delete()
    # delete2()
    delete3()
