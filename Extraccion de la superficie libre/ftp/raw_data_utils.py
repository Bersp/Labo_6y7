import glob
import logging
import os
import warnings
import json

import h5py
import numpy as np
import skimage.io as sio
import yaml

# Logger config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt = '%H:%M:%S')

def get_images_array(images_folder: str, subset: tuple=None):
    images_iterator = sio.imread_collection(images_folder+'*.tif*', conserve_memory=True)
    n = len(images_iterator)

    if subset == None:
        subset = (0, n)
    elif isinstance(subset, int):
        subset = (0, n) if subset > n else (0, subset)
    elif isinstance(subset, tuple):
        subset = (subset[0], n) if subset[1] > n else subset
    else:
        raise 'subset tiene que ser un entero, una tupla o None'

    images_array = np.moveaxis(np.array(images_iterator[subset[0]: subset[1]]),
                               0, 2)
    return images_array

def create_raw_hdf5(med_folder: str, chunks: tuple=(64, 64, 100)):

    # Carpetas en la que busca las imágenes y la información del experimento
    accelerometer_folder = med_folder + 'accelerometer/'
    deformed_folder = med_folder + 'deformed/'
    gray_folder = med_folder + 'gray/'
    reference_folder = med_folder + 'reference/'
    white_folder = med_folder + 'white/'

    # Creo la carpeta HDF5 si no existe y el RAW.HDF5
    hdf5_folder = med_folder + 'HDF5/'
    logging.info(f"RAW: Creando {hdf5_folder+'RAW.hdf5'}")
    if not os.path.isdir(hdf5_folder):
        os.mkdir(hdf5_folder)
    f = h5py.File(hdf5_folder+'RAW.hdf5', 'w')
    logging.info('END\n')

    # Guardo la información de info.yaml
    info_file = med_folder + 'info.yaml'
    logging.info(f'RAW: Agregando atributos de {info_file}')
    with open(info_file, 'r') as stream:
        info = yaml.safe_load(stream)
    for k, v in info.items():
        f.attrs.create(k, json.dumps(v))
    logging.info('END\n')

    # Accelerometer
    logging.info('RAW: Guardando datos del acelerómetro')
    if os.path.isdir(accelerometer_folder):
        data = np.loadtxt(accelerometer_folder+'/acceleration.csv',
                          delimiter=',', unpack=True)
        f.create_dataset('accelerometer', data=data)
    else:
        logging.info(f'No se encontró la carpeta {accelerometer_folder}')
    logging.info('END\n')

    # White
    logging.info('RAW: Guardando imagen blanca')
    if os.path.isdir(white_folder):
        data = sio.imread_collection(white_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('white', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {white_folder}')
    logging.info('END\n')

    # Gray
    logging.info('RAW: Guardando imagen gris')
    if os.path.isdir(gray_folder):
        data = sio.imread_collection(gray_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('gray', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {gray_folder}')
    logging.info('END\n')

    # Reference
    logging.info('RAW: Guardando imagen de referencia')
    if os.path.isdir(reference_folder):
        data = sio.imread_collection(reference_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('reference', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {reference_folder}')
    logging.info('END\n')

    # Deformed
    logging.info('RAW: Guardando imágenes de la superficie deformada')
    if os.path.isdir(deformed_folder):
        resolution = info['CAMERA']['resolution']
        n_images = info['CAMERA']['n_deformed_images']
        deformed_dset = f.create_dataset('deformed',
                                         shape=(*resolution, n_images),
                                         chunks=chunks,
                                         dtype='uint16')
        img_per_chunk = chunks[2]
        n_chunks = np.ceil(n_images/img_per_chunk).astype(int)

        for i in range(n_chunks-1):
            chunk = (img_per_chunk*i, img_per_chunk*(i+1))
            image_chunk = get_images_array(deformed_folder, chunk),
            deformed_dset[:, :, chunk[0]:chunk[1]] = image_chunk
            logging.info(f'RAW: {i+1}/{n_chunks} chunks guardados')

        # Guardo el último chunk aparte porque podría ser más corto
        chunk = (img_per_chunk*(n_chunks-1), n_images)
        deformed_dset[:, :, chunk[0]:chunk[1]] = get_images_array(deformed_folder, chunk)
        logging.info(f'RAW: {n_chunks}/{n_chunks} chunks guardados')

    else:
        logging.info(f'RAW: No se encontró la carpeta {deformed_folder}')
    logging.info('RAW: END\n')

if __name__ == '__main__':
    med_folder = '../../Mediciones_FaradayWaves/MED12 - 0730/'
    create_raw_hdf5(med_folder)
