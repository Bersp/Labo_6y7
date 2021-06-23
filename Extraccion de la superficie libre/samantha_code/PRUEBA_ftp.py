# -*- coding: utf-8 -*-
"""
Prueba a ver que es lo que anda mal. Divido el procesado ftp en dos partes.
"""

import os
import h5py
import numpy as np
import numpy.ma as ma

from input_output import read_parameter_file
from utils import generate_average_image, generate_mask
from calibration import calibrate_camera, undistort_image
from fringe_extrapolation import gerchberg2d
from ftp import calculate_phase_diff_map_1D, height_map_from_phase_map

from matplotlib import cm
import matplotlib.pyplot as plt




"""
Process a RAW datafile by FTP using the parameters from parameter_file.
"""
# Parse kwargs
#if 'destination_path' in kwargs:
#    destination_path = kwargs['destination_path']
#else:
#    destination_path = os.getcwd()

# Read ftp_processing_parameters.yaml parameter file
print('Reading ftp_processing_parameters')
ftp_proc_parameters = read_parameter_file('C:/Users/Samantha/Dropbox/2018_Kucher_Samantha/codigos/repositorio/processing_parameters.yaml')
# Parameters for FTP filtering
n          = ftp_proc_parameters['FTP_PROCESSING']['n']
th         = ftp_proc_parameters['FTP_PROCESSING']['th']
N_iter_max = ftp_proc_parameters['FRINGE_EXTRAPOLATION']['N_iter_max']

# 0(a). Open original datafile and generate pointers to each relevant dataset.
hdf5_input_file    = h5py.File('C:/Users/Samantha/Documents/Facultad/Laboratorio_6/Mediciones/2018-07-17-0001-RAW.hdf5', 'r')
input_dataset_name = hdf5_input_file.attrs['data_series_name']

dset_calib = hdf5_input_file['calibration_camera']
ftp_grp    = hdf5_input_file['ftp_images']
dset_ftp_g = hdf5_input_file['ftp_images/gray']
dset_ftp_r = hdf5_input_file['ftp_images/reference']
dset_ftp_d = hdf5_input_file['ftp_images/deformed']

# Parameters for height calculation
L          = ftp_grp.attrs['L']
D          = ftp_grp.attrs['D']
pspp       = ftp_grp.attrs['pspp']
N_images   = ftp_grp.attrs['N_images']
Slin, Scol = ftp_grp.attrs['shape'] 

corners     = dset_calib.attrs['corners']
square_size = dset_calib.attrs['square_size'] 
# units       = dset_calib.attrs['units']

# 0(b). Create destination hdf5 file and its internal structure.
print('Creating destination hdf5 file')
hdf5_output_filename = input_dataset_name + '-PRO' + '.hdf5'
hdf5_complete_filename_and_path = os.path.join('C:/Users/Samantha/Documents/Facultad/Laboratorio_6/Mediciones/', hdf5_output_filename)
hdf5_output_file   = h5py.File(hdf5_complete_filename_and_path, 'w')
# cal_results_dset   = hdf5_output_file.create_dataset('calibration_results')
height_grp         = hdf5_output_file.create_group('height_fields')
# rectangle_dset     = hdf5_output_file.create_dataset('height_fields/rectangle') 
    
# FTP PROCESSING
# 1. Calibrate camera.
print('Calibrating camera')
cam_mtx, roi, mapx, mapy = calibrate_camera(dset_calib, [8, 6], square_size)

# 2. Generate one gray image by averaging gray images.
print('Generate averages gray and reference, and generate mask')
gray = generate_average_image(dset_ftp_g)

# 3. Generate one reference image by averaging references.
ref  = generate_average_image(dset_ftp_r)

# 4. Undistort gray image.
# gray = undistort_image(gray, mapx, mapy)

# 5. From gray image, determine mask and disk and rectangle properties
mask, c_disk, R_disk, c_rect, sl_rect, mask_of_disk_alone, mask_of_rect_alone = generate_mask(gray)

N_vertical_slices = int(N_images/100) + (1 - (N_images/100).is_integer() )
height_fields_dset = hdf5_output_file.create_dataset('height_fields/disk', \
        shape=(int(2*R_disk), int(2*R_disk), N_images), chunks=(64, 64, N_vertical_slices), dtype='float64')
    
# 6. Undistort reference image.
# ref  = undistort_image(ref, mapx, mapy)

resfactor = np.mean(ref*mask)/np.mean(gray*mask)

# 7. Generate (referece-gray) image.
ref_m_gray = ref - resfactor*gray

# 8. Extrapolate reference image
ref_m_gray  = gerchberg2d(ref_m_gray, mask, N_iter_max=N_iter_max)

# 9. FTP for the deformed dataset
N_defs = dset_ftp_d.shape[2]

dphase_rect = np.zeros(N_defs)
dphase_cent = np.zeros(N_defs)
#%%
# FTP processing of all files by the vertical chunksize previously specified. 
kk = 0
lin_min_idx = int(c_disk[1] - R_disk)
lin_max_idx = int(c_disk[1] + R_disk) 
col_min_idx = int(c_disk[0] - R_disk)
col_max_idx = int(c_disk[0] + R_disk) 

print('FTP processing')
acum = np.zeros((int(2*R_disk), int(2*R_disk), N_vertical_slices))
#%%
for j in range(int(N_images/N_vertical_slices)):
    for i in range(N_vertical_slices):
        # Individual FTP processing follows            
        # 0. Get current image
        def_image = dset_ftp_d[:, :, kk]
        # 1. Undistort image
        # def_image = undistort_image(def_image, mapx, mapy)
        # 2. Substract gray
        def_m_gray = def_image - resfactor*gray        
        # 3. Extrapolate fringes 
        def_m_gray = gerchberg2d(def_m_gray, mask, N_iter_max=N_iter_max)

        # 4. Process by FTP
        dphase = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, \
            th, n)

        dphase_rect[i] = np.mean( dphase[(c_rect[0]-sl_rect):(c_rect[0]+sl_rect), \
               (c_rect[1]-sl_rect):(c_rect[1]+sl_rect)] )

        # 5a). Time unwrapping using rectangle area

        # 5b). Substract solid movement  
        dphase = dphase - dphase_rect[i]
            
        # 6. Calculate height field
        # height_map_from_phase_map(dphase, L, D, pspp)

        # 7. Apply original circular mask
        # dd = ma.masked_array(dphase, mask=(1-mask_of_disk_alone))

        # 8. Crop height field to square 
            
        # 9. Store as a slice in 3D dataset
        # Get data from masked array
        acum[:, :, i] = (ma.getdata(dphase)*mask_of_disk_alone)[lin_min_idx:lin_max_idx,\
                    col_min_idx:col_max_idx]
        kk += 1
            
    height_fields_dset[:, :, ((N_vertical_slices*j)):(N_vertical_slices*(j+1))] = acum 

    
hdf5_output_file.flush()

# The hdf5_output_file should inherit some of the processing parameters as
#  attributes of the root group
    
height_fields_dset.attrs['L'] = L 
height_fields_dset.attrs['D'] = D 
height_fields_dset.attrs['pspp'] = pspp 

height_fields_dset.attrs['freq'] = N_images 
height_fields_dset.attrs['N_images'] = N_images 
    
height_fields_dset.attrs['th'] = th 
height_fields_dset.attrs['n'] = n 

height_fields_dset.attrs['N_iter_max'] = N_iter_max
    
    
    
# Close the input and output files
hdf5_input_file.close()
hdf5_output_file.close()
    