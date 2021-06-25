import numpy as np
import h5py
from matplotlib import pyplot as plt
from unwrap import unwrap
from modulation_instability import bulk_processing
from modulation_instability import input_output
#  from modulation_instability import analysis
import time


destination_path = '/home/bersp/Documents/'
#  input_output.create_original_dataset(
            #  destination_path=destination_path,
            #  parameter_file='datos_para_mandar/capture_parameters.yaml',
            #  cam_calib_path='datos_para_mandar/calibration/',
            #  acc_calib_path='datos_para_mandar/calibration/',
            #  ftp_path='datos_para_mandar/',
            #  deformed_folder='deformed/',
            #  accel_file='')

bulk_processing.process_datafile_by_ftp(
        destination_path=destination_path,
        data_series_file='2018-07-17-0001-RAW.hdf5',
        parameter_file='datos_para_mandar/processing_parameters.yaml'
        )
