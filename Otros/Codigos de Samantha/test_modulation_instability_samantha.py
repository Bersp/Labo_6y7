import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
#  from unwrap import unwrap
#  from modulation_instability import bulk_processing
#  from modulation_instability import input_output
#  from modulation_instability import analysis
#  import time


med_folder = '../../Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/'

processing_parameter_file = med_folder + 'processing_parameters.yaml'
destination_path = med_folder + 'HDF5/'
capture_parameter_file = med_folder + 'capture_parameters.yaml'
cam_calib_path = med_folder + 'calibration/'
acc_calib_path = med_folder + 'calibration/'
ftp_path = med_folder
deformed_folder = 'deformed/'
accel_file = None


#  input_output.create_original_dataset(
            #  destination_path=destination_path,
            #  parameter_file=parameter_file,
            #  cam_calib_path=cam_calib_path,
            #  acc_calib_path=acc_calib_path,
            #  ftp_path=ftp_path,
            #  deformed_folder=deformed_folder,
            #  accel_file=accel_file)

#  data_series_file = '/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-RAW.hdf5'
#  processing_parameter_file = med_folder + 'processing_parameters.yaml'

#  bulk_processing.process_datafile_by_ftp(
        #  destination_path=destination_path,
        #  data_series_file=data_series_file,
        #  parameter_file=processing_parameter_file)

#  f = h5py.File('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5', 'r')
f = h5py.File('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2019-10-09-a200-f20-fmod0-annulus-PRO.hdf5', 'r')

annulus_deformed = f['height_fields/annulus']
annulus_mask = f['masks/annulus']

#  levels = np.min(annulus_deformed[:, :, 0:20]), np.max(annulus_deformed[:, :, 0:20])
#  levels = np.linspace(levels[0], levels[1], 300)
#  colors = plt.get_cmap('gray')(np.linspace(0, 1, 300))[:-1]

#  cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
#  print(cmap)
