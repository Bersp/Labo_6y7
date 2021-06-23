import os
import h5py
import numpy as np
import numpy.ma as ma
import yaml
import skimage.io as sio

from modulation_instability.input_output import read_parameter_file
from modulation_instability.utils import generate_average_image, generate_mask
#from modulation_instability.calibration import calibrate_camera, undistort_image
from modulation_instability.fringe_extrapolation import gerchberg2d
from modulation_instability.ftp import calculate_phase_diff_map_1D, height_map_from_phase_map
from modulation_instability.parallelized import individual_ftp

from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()

from multiprocessing import Pool, cpu_count
from itertools import repeat

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

cores = cpu_count()
p = Pool(cores)


def process_datafile_by_ftp(*, data_series_file, parameter_file, **kwargs):
    """
    Process a RAW datafile by FTP using the parameters from parameter_file.
    """
    # Parse kwargs
    if 'destination_path' in kwargs:
        destination_path = kwargs['destination_path']
    else:
        destination_path = os.getcwd()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Read ftp_processing_parameters.yaml parameter file
    logger.info('Reading ftp_processing_parameters')
    ftp_proc_parameters = read_parameter_file(parameter_file)
    # Parameters for FTP filtering
    n          = ftp_proc_parameters['FTP_PROCESSING']['n']
    th         = ftp_proc_parameters['FTP_PROCESSING']['th']
    N_iter_max = ftp_proc_parameters['FRINGE_EXTRAPOLATION']['N_iter_max']

    # 0(a). Open original datafile and generate pointers to each relevant dataset.
    hdf5_input_file    = h5py.File(data_series_file, 'r')
    input_dataset_name = hdf5_input_file.attrs['data_series_name']

    #dset_calib = hdf5_input_file['calibration_camera']
    ftp_grp    = hdf5_input_file['ftp_images']
    dset_ftp_g = hdf5_input_file['ftp_images/gray']
    dset_ftp_r = hdf5_input_file['ftp_images/reference']
    dset_ftp_d = hdf5_input_file['ftp_images/deformed']

    # Parameters for height calculation
    L          = ftp_grp.attrs['L']
    D          = ftp_grp.attrs['D']
    pspp       = ftp_grp.attrs['pspp']
    N_images   = dset_ftp_d.shape[2] #ftp_grp.attrs['N_images']
    Slin, Scol = ftp_grp.attrs['shape']

    #corners     = dset_calib.attrs['corners']
    #square_size = dset_calib.attrs['square_size']
    # units       = dset_calib.attrs['units']

    # 0(b). Create destination hdf5 file and its internal structure.
    logger.info('Creating destination hdf5 file')
    # hdf5_output_filename_disk = input_dataset_name + '-disk-PRO' + '.hdf5'
    hdf5_output_filename_annulus = input_dataset_name + '-annulus-PRO' + '.hdf5'
    # hdf5_complete_filename_and_path_disk = os.path.join(destination_path, hdf5_output_filename_disk)
    hdf5_complete_filename_and_path_annulus = os.path.join(destination_path, hdf5_output_filename_annulus)
    # hdf5_output_file_disk   = h5py.File(hdf5_complete_filename_and_path_disk, 'w')
    hdf5_output_file_annulus   = h5py.File(hdf5_complete_filename_and_path_annulus, 'w')
    # cal_results_dset   = hdf5_output_file.create_dataset('calibration_results')
    # height_grp_disk         = hdf5_output_file_disk.create_group('height_fields')
    height_grp_annulus      = hdf5_output_file_annulus.create_group('height_fields')
    # mask_grp_disk     = hdf5_output_file_disk.create_group('masks')
    mask_grp_annulus     = hdf5_output_file_annulus.create_group('masks')


    # FTP PROCESSING
    # 1. Calibrate camera.
    #logger.info('Calibrating camera')
    #cam_mtx, roi, mapx, mapy = calibrate_camera(dset_calib, [8, 6], square_size)

    # 2. Generate one gray image by averaging gray images.
    logger.info('Generate averages gray and reference, and generate mask')
    #period = 5

    gray = generate_average_image(dset_ftp_g)

    # 3. Generate one reference image by averaging references.
    ref  = generate_average_image(dset_ftp_r)

    # 4. Undistort gray image.
    # gray = undistort_image(gray, mapx, mapy)

    # 5. From gray image, determine mask and disk, rectangle and annulus properties

    ###   AGREGO ASI NOMAS LA MASCARA ###
    mask, c_disk, R_disk, c_rect, sl_rect, c_annulus, R_ext_annulus, R_int_annulus, mask_of_disk_alone, mask_of_rect_alone, mask_of_annulus_alone = generate_mask(gray, modificado=True)
    binary_img = mask.copy()
    yc, xc = c_disk
    '''
    Rint, Rext, Rgrande = R_disk, R_int_annulus-4, R_ext_annulus
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if ((i-xc)**2+(j-yc)**2>Rint**2) and ((i-xc)**2+(j-yc)**2<Rext**2) and (j-yc)>0 and (i-xc)>0:
                binary_img[i,j] = 0
            if ((i-xc)**2+(j-yc)**2>Rgrande**2) and (j-yc)>0 and (i-xc)>0:
                binary_img[i,j] = 0
    mask, c_disk, R_disk, c_rect, sl_rect, c_annulus, R_ext_annulus, R_int_annulus, mask_of_disk_alone, mask_of_rect_alone, mask_of_annulus_alone = generate_mask(binary_img)
    Rint, Rext, Rgrande = R_disk, R_int_annulus-15, R_ext_annulus+8
    '''
    Rint, Rext, Rgrande = R_disk, R_int_annulus-8, R_ext_annulus+4
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if ((i-xc)**2+(j-yc)**2>Rext**2) and ((i-xc)**2+(j-yc)**2<Rgrande**2):
                binary_img[i,j] = 1

    mask_g, c_disk_g, R_disk_g, c_rect_g, sl_rect_g, c_annulus_g, R_ext_annulus_g, R_int_annulus_g, mask_of_disk_alone_g, mask_of_rect_alone_g, mask_of_annulus_alone_g = generate_mask(binary_img)

    N_vertical_slices = dset_ftp_d.chunks[2]

    # height_fields_disk_dset = hdf5_output_file_disk.create_dataset('height_fields/disk', shape=(int(2*R_disk), int(2*R_disk), N_images), chunks=(64, 64, N_vertical_slices), dtype='float64')
    height_fields_annulus_dset = hdf5_output_file_annulus.create_dataset('height_fields/annulus', \
            shape=(int(2*R_ext_annulus), int(2*R_ext_annulus), N_images), chunks=(64, 64, N_vertical_slices), dtype='float64')

    # 6. Undistort reference image.
    # ref  = undistort_image(ref, mapx, mapy)

    resfactor = np.mean(ref*mask)/np.mean(gray*mask)

    # 7. Generate (referece-gray) image.
    ref_m_gray = ref - resfactor*gray

    # 8. Extrapolate reference image
    ref_m_gray  = gerchberg2d(ref_m_gray, mask_g, N_iter_max=N_iter_max)

    # 9. FTP for the deformed dataset
    # pongo que haga menos a ver como da
    N_defs = dset_ftp_d.shape[2]

    # dphase_rect = np.zeros(N_defs)

    # FTP processing of all files by the vertical chunksize previously specified.
    # kk = 0

    lin_min_idx_a = abs(int(c_annulus[1] -R_ext_annulus))
    lin_max_idx_a = lin_min_idx_a + int(2*R_ext_annulus) #int(c_annulus[1] + R_ext_annulus)
    col_min_idx_a = abs(int(c_annulus[0] - R_ext_annulus))
    col_max_idx_a = col_min_idx_a + int(2*R_ext_annulus) #int(c_annulus[0] + R_ext_annulus)
    #ACÁ ESTÁ EL ERROR!
    #lin_max_idx - lin_min_idx deberia ser igual a int(2*R_ext_annulus)

    # lin_min_idx_d = (int(c_disk[1] - R_disk))
    # lin_max_idx_d = lin_min_idx_d + int(2*R_disk) #int(c_disk[1] + R_disk)
    # col_min_idx_d = int(c_disk[0] - R_disk)
    # col_max_idx_d = col_min_idx_d + int(2*R_disk) #int(c_disk[0] + R_disk)

    logger.info('FTP processing')
    acum_a = np.zeros((int(2*R_ext_annulus), int(2*R_ext_annulus), N_vertical_slices))
    # acum_d = np.zeros((int(2*R_disk), int(2*R_disk), N_vertical_slices))
    #pp=0

    for j in range(int(N_images/N_vertical_slices)):
        print(j)
        deformed = dset_ftp_d[:,:,((N_vertical_slices*j)):(N_vertical_slices*(j+1))]
        non_iterable_args = deformed, resfactor, gray, mask_g, ref_m_gray, N_iter_max, n, th, c_rect, sl_rect, mask_of_annulus_alone, lin_min_idx_a, lin_max_idx_a, col_min_idx_a, col_max_idx_a
        height = p.starmap(individual_ftp, zip(np.arange(N_vertical_slices), repeat(non_iterable_args)))

            # 9. Store as a slice in 3D dataset
            # Get data from masked array
        # acum_a = (ma.getdata(height)*mask_of_annulus_alone)[lin_min_idx_a:lin_max_idx_a, col_min_idx_a:col_max_idx_a]
            # acum_d[:, :, i] = (ma.getdata(height)*mask_of_disk_alone)[lin_min_idx_d:lin_max_idx_d, col_min_idx_d:col_max_idx_d]
            # kk += 1

        height_fields_annulus_dset[:, :, ((N_vertical_slices*j)):(N_vertical_slices*(j+1))] = np.dstack(height)
        # height_fields_disk_dset[:, :, ((N_vertical_slices*j)):(N_vertical_slices*(j+1))] = acum_d


    # hdf5_output_file_disk.flush()
    hdf5_output_file_annulus.flush()


    #Store masks in two datasets
    # mask_disk_dset = hdf5_output_file_disk.create_dataset('masks/disk', shape=(int(2*R_disk), int(2*R_disk), 1), dtype='float64')
    mask_annulus_dset = hdf5_output_file_annulus.create_dataset('masks/annulus', \
            shape=(int(2*R_ext_annulus), int(2*R_ext_annulus), 1), dtype='float64')

    #mask_disk_dset[:,:,0] = mask_of_disk_alone[lin_min_idx_d:lin_max_idx_d, col_min_idx_d:col_max_idx_d]
    mask_annulus_dset[:,:,0] = mask_of_annulus_alone[lin_min_idx_a:lin_max_idx_a,\
                    col_min_idx_a:col_max_idx_a]
    # The hdf5_output_file should inherit some of the processing parameters as
    #  attributes of the root group

#     height_fields_disk_dset.attrs['L'] = L
#     height_fields_disk_dset.attrs['D'] = D
#     height_fields_disk_dset.attrs['pspp'] = pspp
# 
#     height_fields_disk_dset.attrs['freq'] = N_images
#     height_fields_disk_dset.attrs['N_images'] = N_images
# 
#     height_fields_disk_dset.attrs['th'] = th
#     height_fields_disk_dset.attrs['n'] = n
# 
#     height_fields_disk_dset.attrs['N_iter_max'] = N_iter_max
# 


    # Close the input and output files
    hdf5_input_file.close()
    # hdf5_output_file_disk.close()
    hdf5_output_file_annulus.close()


    return None
