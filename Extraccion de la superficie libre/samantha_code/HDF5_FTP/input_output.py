import os
import glob
import h5py
import numpy as np
import skimage.io as sio
import yaml
from modulation_instability.utils import divisor

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


def create_original_dataset(*, parameter_file, cam_calib_path, acc_calib_path, ftp_path, deformed_folder, accel_file, **kwargs):
    """
    Creates an original dataset HDF5 file. This function should be used by user instead of generate_original_dataset.
    """
    # Parse kwargs
    if 'destination_path' in kwargs:
        destination_path = kwargs['destination_path']
    else:
        destination_path = os.getcwd()

    # Parse parameter file to determine name of destination file
    name_for_hdf5_file = read_parameter_file(parameter_file)['DATA_SERIES_NAME'] + '-RAW' + '.hdf5'
    complete_path_and_name_for_hdf5_file = os.path.join(destination_path, name_for_hdf5_file)
    # Generate the dataset
    generate_original_dataset(complete_path_and_name_for_hdf5_file, cam_calib_path, acc_calib_path, ftp_path, deformed_folder, accel_file, parameter_file)
    return None


def generate_original_dataset(hdf5_datasetname, cam_calib_path, acc_calib_path, ftp_im_path, deformed_folder, accel_file, parameter_file, **kwargs):
    """
    Generates an original dataset HDF5 file according to the following scheme.
    """
    # HDF5 file creation
    hdf5_file = h5py.File(hdf5_datasetname, 'w')

    # Read capture_parameters.yaml file
    parameters = read_parameter_file(parameter_file)

    # Assign image size
    Slin, Scol = parameters['CAMERA_CAPTURE']['shape']
    logger.info('Creando el dataset de la camara...')
    # Creation of calibration_camera group
    # If cam_calib_path is an actual path, then store individual images in a dataset
    # If cam_calib_path is a file (a HDF5 file), then grab calibration results from it
    if os.path.isdir(cam_calib_path):
        cal_files = sorted(glob.glob(os.path.join(cam_calib_path, '*.tiff')))
        Ncal = len(cal_files)
        cal_dset = hdf5_file.create_dataset('calibration_camera', \
                shape=(Slin, Scol, Ncal), chunks=(64, 64, Ncal), dtype='float64')

        ii = 0
        for file in cal_files:
            image = sio.imread(file)
            image = image.astype(float)
            cal_dset[:, :, ii] = image
            ii += 1

    elif os.path.isfile(cam_calib_path):
        # grab camera calibration results from file
        pass
    else:
        pass


    hdf5_file.flush()

    # Creation of ftp_images group
    ftp_grp = hdf5_file.create_group('ftp_images')
    logger.info('Creando dataset de grises')
    # Create gray dataset
    gri_files = sorted(glob.glob(os.path.join(ftp_im_path, 'gray', '*.tiff')))
    Ngri = len(gri_files)
    N_gri_vertical_slices = divisor(Ngri)
    gri_dset = hdf5_file.create_dataset('ftp_images/gray', \
            shape=(Slin, Scol, Ngri), chunks=(64, 64, N_gri_vertical_slices), dtype='float64')
    kk=0
    acum_gri = np.zeros((Slin, Scol, N_gri_vertical_slices))
    for j in range(int(Ngri/N_gri_vertical_slices)):
        for i in range(N_gri_vertical_slices):
            image = sio.imread(gri_files[kk])
            image = image.astype(float)
            acum_gri[:, :, i] = image
            
            kk += 1

        gri_dset[:, :, ((N_gri_vertical_slices*j)):(N_gri_vertical_slices*(j+1))] = acum_gri

    hdf5_file.flush()

    logger.info('Creando dataset de referencia')
    # Create reference dataset
    ref_files = sorted(glob.glob(os.path.join(ftp_im_path , 'reference', '*.tiff')))
    Nref = len(ref_files)
    N_ref_vertical_slices = divisor(Nref)
    ref_dset = hdf5_file.create_dataset('ftp_images/reference', \
            shape=(Slin, Scol, Nref), chunks=(64, 64, N_ref_vertical_slices), dtype='float64')
    kk=0
    acum_ref = np.zeros((Slin, Scol, N_ref_vertical_slices))
    for j in range(int(Nref/N_ref_vertical_slices)):
        for i in range(N_ref_vertical_slices):
            image = sio.imread(ref_files[kk])
            image = image.astype(float)
            acum_ref[:, :, i] = image

            kk += 1

        ref_dset[:, :, ((N_ref_vertical_slices*j)):(N_ref_vertical_slices*(j+1))] = acum_ref


    hdf5_file.flush()

    logger.info('Creando dataset deformadas')
    # Create deformed dataset
    def_files = sorted(glob.glob(os.path.join(ftp_im_path , deformed_folder, '*.tiff')))
    Ndef = len(def_files)

    N_vertical_slices = divisor(Ndef)

    def_dset = hdf5_file.create_dataset('ftp_images/deformed', \
            shape=(Slin, Scol, Ndef), chunks=(64, 64, N_vertical_slices), dtype='float64')

    kk = 0
    acum = np.zeros((Slin, Scol, N_vertical_slices))
    for j in range(int(Ndef/N_vertical_slices)):
        for i in range(N_vertical_slices):
            image = sio.imread(def_files[kk])
            image = image.astype(float)
            acum[:, :, i] = image
            kk += 1

        def_dset[:, :, ((N_vertical_slices*j)):(N_vertical_slices*(j+1))] = acum

    hdf5_file.flush()
    logger.info('Listo.')

    # accelerometer_readings_dset = hdf5_file.create_dataset('raw_accelerometer_readings')

    # Creation of calibration_accelerometer group
    # If acc_calib_path is an actual path, then store individual images in a dataset
    # If acc_calib_path is a file (a HDF5 file), then grab calibration results from it
    if os.path.isdir(acc_calib_path):
        # Store accelerometer readings
        # which contains the accelerometer_readings as a numpy array with
        # 4 columns (time, acc_x, acc_y, acc_z), and as many rows as readings
        pass
    elif os.path.isfile(acc_calib_path):
        # Grab accelerometer calibration results from HDF5 file
        pass


    hdf5_file.flush()


    # Storing the capture_parameters dataset as attributes to each dataset
    # Data series name, forcing type, forcing & modulating frequencies + comments
    hdf5_file.attrs['data_series_name'] = parameters['DATA_SERIES_NAME']
    hdf5_file.attrs['type'] = parameters['FORCING']['type']
    hdf5_file.attrs['osc_freq'] = parameters['FORCING']['osc_freq']
    hdf5_file.attrs['mod_freq'] = parameters['FORCING']['mod_freq']
    hdf5_file.attrs['comments'] = parameters['COMMENTS']
    # Camera calibration parameters
    #cal_dset.attrs['corners'] = parameters['CAMERA_CALIBRATION']['corners']
    #cal_dset.attrs['square_size'] = parameters['CAMERA_CALIBRATION']['square_size']
    #cal_dset.attrs['units'] = parameters['CAMERA_CALIBRATION']['units']
    # FTP Projection parameters
    ftp_grp.attrs['L'] = parameters['FTP_PROJECTION']['L']
    ftp_grp.attrs['D'] = parameters['FTP_PROJECTION']['D']
    ftp_grp.attrs['pspp'] = parameters['FTP_PROJECTION']['pspp']
    ftp_grp.attrs['type'] = parameters['FTP_PROJECTION']['type']
    ftp_grp.attrs['units'] = parameters['FTP_PROJECTION']['units']
    # Camera capture parameters
    ftp_grp.attrs['shape'] = parameters['CAMERA_CAPTURE']['shape']
    ftp_grp.attrs['camera_freq'] = parameters['CAMERA_CAPTURE']['freq']
    ftp_grp.attrs['N_images'] = parameters['CAMERA_CAPTURE']['N_images']
    # Accelerometer parameters
    ftp_grp.attrs['accelerometer_freq'] = parameters['ACCELEROMETER_CAPTURE']['freq']

    hdf5_file.flush()

    hdf5_file.close()


def read_parameter_file(yamlfile):
    """
    Reads a YAML parameter file, returning it as a dictionary."
    """
    return yaml.load(open(yamlfile))


def create_empty_parameter_file(path_for_file):
    """
    Creates empty parameter file to accompany an original dataset.
    """
