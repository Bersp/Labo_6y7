import glob
import json
import logging
import os
import sys
import warnings

import h5py
from mpi4py import MPI
import numpy as np
import skimage.io as sio
import yaml
num_processes = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank

# Logger config
logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s (CORE: {rank}) | %(message)s',
                    datefmt='%H:%M:%S')


def get_images_array(images_folder: str, subset: tuple = None):
    """
    TODO: Docstring for get_images_array
    """
    images_iterator = sio.imread_collection(
        images_folder+'*.tif*', conserve_memory=True)
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


def create_raw_hdf5(med_folder: str, chunks: tuple = (64, 64, 100)):
    """
    TODO: Docstring for create_raw_hdf5
    """

    hdf5_folder = med_folder + 'HDF5/'
    # TODO: Eliminar esta parte. Las carpetas HDF5 se crean desde antes.
    # Creamos la carpeta HDF5 si no existe (la crea el primer nucleo que llega)
    # if not os.path.isdir(hdf5_folder):
    # logging.info(f'RAW: Creando {hdf5_folder}')
    # os.mkdir(hdf5_folder)

    # Creamos el RAW.HDF5 (h5py se encarga de paralelizar bien)
    logging.info(f"RAW: Abriendo {hdf5_folder+'RAW.hdf5'}")
    f = h5py.File(hdf5_folder+'RAW.hdf5', 'w',
                  driver='mpio', comm=MPI.COMM_WORLD)

    # La ubicación de deformed_folder la necesitan todos los nucleos
    deformed_folder = med_folder + 'deformed/'

    # La información sobre las mediciones la necesitan todos los nucleos
    info_file = med_folder + 'info.yaml'
    with open(info_file, 'r') as stream:
        info = yaml.safe_load(stream)

    # Carpetas en la que busca las imágenes y la información del experimento
    accelerometer_folder = med_folder + 'accelerometer/'
    gray_folder = med_folder + 'gray/'
    reference_folder = med_folder + 'reference/'
    white_folder = med_folder + 'white/'

    # NOTE: El proceso de guardado de info.yaml, accelerometer, white, gray y reference
    #       se repite para cada nucleo, esto es necesario dado que h5py necesita conocer
    #       el total de datasets creados
    logging.info(
        f'RAW: Guardando info.yaml, accelerometer, white, gray y reference')

    # Guardo la información de info.yaml
    for k, v in info.items():
        f.attrs.create(k, json.dumps(v))

    # Accelerometer
    if os.path.isdir(accelerometer_folder):
        data = np.loadtxt(accelerometer_folder+'/acceleration.csv',
                          delimiter=',', unpack=True)
        f.create_dataset('accelerometer', data=data)
    else:
        logging.info(f'No se encontró la carpeta {accelerometer_folder}')

    # White
    if os.path.isdir(white_folder):
        data = sio.imread_collection(white_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('white', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {white_folder}')

    # Gray
    if os.path.isdir(gray_folder):
        data = sio.imread_collection(gray_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('gray', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {gray_folder}')

    # Reference
    if os.path.isdir(reference_folder):
        data = sio.imread_collection(reference_folder+'*.tif*',
                                     conserve_memory=True)
        data = np.asarray(data).mean(axis=0)
        f.create_dataset('reference', data=data)
    else:
        logging.info(f'RAW: No se encontró la carpeta {reference_folder}')

    # Deformed. Todos los nucleos saben de la creación del dataset
    #           pero cada uno escribre chunks de forma independiente
    logging.info(f'RAW: Guardando imágenes de la superficie deformada')
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
            if i % num_processes == rank:
                chunk = (img_per_chunk*i, img_per_chunk*(i+1))
                image_chunk = get_images_array(deformed_folder, chunk),
                deformed_dset[:, :, chunk[0]:chunk[1]] = image_chunk
                logging.info(f'RAW: {i+1}/{n_chunks} chunks guardados')

        # Guardo el último chunk aparte porque podría ser más corto
        if rank == 0:
            chunk = (img_per_chunk*(n_chunks-1), n_images)
            deformed_dset[:, :, chunk[0]:chunk[1]
                          ] = get_images_array(deformed_folder, chunk)
            logging.info(f'RAW: {n_chunks}/{n_chunks} chunks guardados')
    else:
        logging.info(f'RAW: No se encontró la carpeta {deformed_folder}')

    logging.info(f'RAW: Esperando para cerrar el HDF5')
    f.close()
    logging.info(f'RAW: END')


def main():
    # med_folder = '../../Mediciones/MED40 - Oscilones a full - 0902/'
    med_folder = sys.argv[1]
    create_raw_hdf5(med_folder, chunks=(64, 64, 20))


if __name__ == "__main__":
    main()
