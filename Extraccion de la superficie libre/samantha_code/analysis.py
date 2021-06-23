import os
import h5py
import time
import numpy as np
from scipy.signal import hilbert
from PyEMD import EMD
import scipy.io as sio
from modulation_instability.input_output import read_parameter_file
from modulation_instability.utils import polar2cart, img2polar, sum_values, find_first_and_last, find_index_of_false_values, divisor
from modulation_instability.parallelized import polar_coordinates
from scipy.special import erf
import numpy.ma as ma

from multiprocessing import Pool, cpu_count
from itertools import repeat

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

cores = cpu_count()
p = Pool(cores)

def load_parameters(dset_in):
    """
    Returns useful parameters, like shape in every direction and the vertical length of a chunk.
    """
    N = dset_in.shape[2]
    N_vertical_slices = dset_in.chunks[2]
    Slin, Scol = dset_in.shape[0], dset_in.shape[1]//2 #check!
    return N, N_vertical_slices, Slin, Scol

def save_fourier_transform(*,data_file, analysis_file):
    """
    Performs spatial and temporal Fourier transform on a 3D dataset. Executes 'transform_fourier_2d_slices' and 'transform_fourier_temporal'.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Creating hdf5 output file')
    f_in = h5py.File(data_file, 'r')
    dset_data = f_in['height_fields/disk']
    f_out = h5py.File(analysis_file, 'w')
    N, N_vertical_slices, Slin, Scol = load_parameters(dset_data)

    # 2D Fourier Transform
    logger.info('Performing 2D Fourier Transform')
    dset_out_2d = f_out.create_dataset('fft2D', shape=(Slin, Slin, N), chunks=(64, 64, N_vertical_slices), dtype='c16')
    transform_fourier_2d_slices(dset_data, dset_out_2d, Slin+10)

    # 3D Fourier Transform
    # If it is applied on dset_data is only temporal transform
    logger.info('Performing temporal Fourier Transform')
    dset_out_3d = f_out.create_dataset('fft3D', (Slin, Slin,N), chunks=(dset_data.chunks[0],
    dset_data.chunks[1], N_vertical_slices), dtype='c16')
    transform_fourier_temporal(dset_out_2d, dset_out_3d)

    f_in.close()
    f_out.close()

def save_envelope_carrier(data_file, analysis_file, fs=100):
    """
    Implements the EMD algorithm to every 'column' of a 3D dataset, using 'sum_emds'. Then performs the Hilbert Transform to obtain the envelope and saves the instantaneous frequency and phase.
    fs: (int) Sampling rate
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Creating hdf5 output file')
    f_in = h5py.File(data_file, 'r')
    dset_data = f_in['height_fields/disk']

    f_out = h5py.File(analysis_file, 'w')
    N, N_vertical_slices, Slin, Scol = load_parameters(dset_data)

    # Generate dataset for envelope
    logger.info('Generating datasets fot envelope and carrier')
    dset_out_envelope = f_out.create_dataset('envelope', (Slin, Slin,N), chunks=(dset_data.chunks[0],
    dset_data.chunks[1], N_vertical_slices), dtype='float')

    # Generate dataset for carrier
    dset_out_carrier_phase = f_out.create_dataset('carrier_phase', (Slin, Slin,N), chunks=(dset_data.chunks[0],
    dset_data.chunks[1], N_vertical_slices), dtype='float')
    dset_out_carrier_freq = f_out.create_dataset('carrier_freq', (Slin, Slin,N-1), chunks=(dset_data.chunks[0],
    dset_data.chunks[1], N_vertical_slices), dtype='float')

    #fs = f_in.attrs['freq']
    #fs=100
    logger.info('Performing iterations')
    for i in range(0,Slin, dset_data.chunks[0]):
        if i+dset_data.chunks[0]<Slin:
            last_size_i = dset_data.chunks[0]
        else:
            last_size_i = i+dset_data.chunks[0]-Slin
        print('i=',i)
        for j in range(0, Slin, dset_data.chunks[1]):
           #print('j=',j)
            if j+dset_data.chunks[1]<Slin:
                last_size_j = dset_data.chunks[1]
            else:
                last_size_j = j+dset_data.chunks[1]-Slin
            volume = dset_data[i:(i+last_size_i), j:(j+last_size_j), :]
            new_volume = np.zeros(shape=volume.shape)
            for k in range(0, last_size_i):
               # print('k=',k)
                for l in range(0, last_size_j):
                   # print(l)
                    column = volume[k,l,:]
                    # Empirical Mode Decomposition
                    new_volume[k,l,:] = sum_emds(column, fs)
            # Hilbert Transform
            analytic_signal = hilbert(new_volume)
            # Envelope
            amplitude_envelope = np.abs(analytic_signal)
            dset_out_envelope[i:(i+last_size_i), j:(j+last_size_j), :] = amplitude_envelope
            # Carrier
            phase, freq = return_instantaneous_parameters(volume, fs)
            dset_out_carrier_phase[i:(i+last_size_i), j:(j+last_size_j), :] = phase
            dset_out_carrier_freq[i:(i+last_size_i), j:(j+last_size_j), :] = freq

    f_in.close()
    f_out.close()


def transform_fourier_2d_slices(dset_in, dset_out, matrix_side):
    """
    Performs 2D spatial Fourier Transform on every 'slice' of a 3D dataset.
    """
    N, N_vertical_slices, Slin, Scol = load_parameters(dset_in)
    acum = np.zeros((Slin, Slin, N_vertical_slices), 'complex')

    radius = dset_in.shape[0]//2
    xc, yc = matrix_side//2, matrix_side//2
    xx, yy = np.arange(matrix_side), np.arange(matrix_side)
    (xi, yi) = np.meshgrid(xx,yy)
    error_function = -erf((xi-xc)**2+(yi-yc)**2-radius**2)+1
    new = (matrix_side-dset_in.shape[0])//2

    for j in range(N//N_vertical_slices):
        vol = dset_in[:,:,(N_vertical_slices*j):(N_vertical_slices*(j+1))]
        for i in range(N_vertical_slices):
            matrix = (np.pad(vol[:,:,i], [new, new], mode='constant'))*error_function
            m = np.fft.fftshift(np.fft.fft2(matrix))
            m = m[new: m.shape[0]-new, new: m.shape[1]-new]
            acum[:,:,i] = m
        dset_out[:,:,(N_vertical_slices*j):(N_vertical_slices*(j+1))] = acum
    dset_out.shape
    dset_out.flush()



def transform_fourier_temporal(dset_in, dset_out):
    """
    Performs 1D temporal Fourier Transform on every 'column' of a 3D dataset.
    """
    N, N_vertical_slices, Slin, Scol = load_parameters(dset_in)

    for i in range(0,Slin, dset_in.chunks[0]):
        for j in range(0, Slin, dset_in.chunks[1]):
            tf = np.fft.fft(dset_in[i:(i+dset_in.chunks[0]), j:(j+dset_in.chunks[1]), :])
            dset_out[i:(i+dset_in.chunks[0]), j:(j+dset_in.chunks[1]), :] = tf
    dset_out.flush()


def sum_emds(col, fs):
    """
    Decomposes the signal into 'intrinsic mode functions' (IMF) and sum all IMFs except the last one.
    """
    # Empirical Mode Decomposition
    M = len(col)
    t = np.linspace(0,M/fs,M)
    IMF = EMD().emd(col,t)
    z = np.zeros(IMF.shape[1])
    # Sum all imfs except the last one
    for n in range(IMF.shape[0]-1):
        z = z+IMF[n]
    return z


def return_instantaneous_parameters(col, fs):
    instantaneous_phase = np.unwrap(np.angle(col))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)
    return instantaneous_phase, instantaneous_frequency

# def height_annulus_polar_coordinates(frame, mask, phase_width = 3000):
#     """
#     Returns a matrix with the height of the annulus in polar coordinates.
#     """
#     mask_annulus = mask[:,:,0]
#     annulus = frame
#     annulus_polar = img2polar(ma.masked_array(annulus, mask=(1-mask_annulus)))
#     first_idx, last_idx = find_index_of_false_values(annulus_polar)
#     if first_idx != last_idx:
#         annulus_polar_lines = ma.getdata(annulus_polar[first_idx:last_idx,:]) #por si toma mal la mascara
#         prom = np.average(annulus_polar_lines, axis=0)
#     else:
#         annulus_polar_lines = ma.getdata(annulus_polar[first_idx,:])
#         prom = annulus_polar_lines
#     return prom
# 
def save_height_annulus_polar_coordinates(file):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Opening input file')
    f = h5py.File(file, 'r+')
    dset_in = f['height_fields/annulus']
    mask_annulus = f['masks/annulus']
    mask_annulus = mask_annulus[:,:,0]
    phase_width = 3000

    N = dset_in.shape[2]
    N_vertical_slices = dset_in.chunks[2]

    logger.info('Generating height averages in polar coordinates')
    dset_out = f.create_dataset('height_fields/annulus_polar', (N, phase_width), chunks=(N_vertical_slices, phase_width), dtype='float')

    # kk = 0
    acum = np.zeros((N_vertical_slices, phase_width))
    for j in range(int(N/N_vertical_slices)):
        print(j)
        deformed = dset_in[:, :,((N_vertical_slices*j)):(N_vertical_slices*(j+1))]
        non_iterable_args = deformed, mask_annulus
        acum = p.starmap(polar_coordinates, zip(np.arange(N_vertical_slices), repeat(non_iterable_args)))

#         for i in range(N_vertical_slices):
#             print(kk)
#             #frame = dset_in[:,:,kk]
# 
#             frame = deformed[:,:,i]
#             prom = height_annulus_polar_coordinates(frame, mask_annulus)
#             acum[:,i] = prom
# 
#             kk += 1
# 
        dset_out[((N_vertical_slices*j)):(N_vertical_slices*(j+1)),:] = np.transpose(np.dstack(acum)[0,:,:])
    f.flush()
    f.close()

def save_annulus_fourier_transform(data_file, analysis_file, create_new_file=False):
    """
    Using the height of the annulus in polar coordinates, performs a 2D fourier transform. Rows (matrix.shape[0])  represent 'time' and columns (matrix.shape[1]) represent 'space'.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Opening input file')
    f_in = h5py.File(data_file, 'r')
    dset_data = f_in['height_fields/annulus_polar']
    #mask_annulus = f_in['masks/annulus']

    #logger.info('Generating height averages in polar coordinates')
    #matrix = height_annulus_polar_coordinates(dset_data, mask_annulus)
    logger.info('Performing Fourier Transform')
    m = np.fft.fftshift(np.fft.fft2(dset_data))

    if create_new_file == False:
        logger.info('Opening output file')
        f_out = h5py.File(analysis_file, 'r+')
    else:
        logger.info('Creating output file')
        f_out = h5py.File(analysis_file, 'w')

    logger.info('Opening output file')
    f_out = h5py.File(analysis_file, 'r+')

    logger.info('Generating dataset')
    dset_out = f_out.create_dataset('fft2d_annulus', dset_data.shape, chunks=True, dtype='complex')
    dset_out[:,:] = m

    dset_out.flush()
    f_in.close()
    f_out.close()

def save_envelope_carrier_annulus(data_file, analysis_file, create_new_file=False, fs=100):
    """
    Implements the EMD algorithm to every 'row' of a 2D dataset, using 'sum_emds'. Then performs the Hilbert Transform to obtain the envelope and saves the instantaneous frequency and phase. Saves the results in an existing file.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Opening hdf5 output file')
    f_in = h5py.File(data_file, 'r')
    dset_data = f_in['height_fields/annulus_polar']
    mask_annulus = f_in['masks/annulus']

    logger.info('Generating height averages in polar coordinates')
    matrix = dset_data[:,:]
    if create_new_file == False:
        f_out = h5py.File(analysis_file, 'r+')
    else:
        f_out = h5py.File(analysis_file, 'w')

    # Generate dataset for emd recomposed signal
    logger.info('Generating dataset fot emd recomposed signal')
    dset_out_emd = f_out.create_dataset('emd_recomposed', matrix.shape, chunks=True, dtype='complex128')
    # Generate dataset for envelope
    logger.info('Generating datasets fot envelope and carrier')
    dset_out_envelope = f_out.create_dataset('envelope_annulus', matrix.shape, chunks=True, dtype='float')

    # Generate dataset for carrier
    dset_out_carrier_phase = f_out.create_dataset('carrier_phase_annulus', matrix.shape, chunks=True, dtype='float')
    dset_out_carrier_freq = f_out.create_dataset('carrier_freq_annulus', shape = (matrix.shape[0], matrix.shape[1]-1), chunks=True, dtype='float')

    new_matrix = np.zeros(shape=matrix.shape)
    logger.info('Performing iterations')
    for i in range(matrix.shape[0]):
      new_matrix[i,:] = sum_emds(matrix[i,:], fs)
    # Hilbert Transform
    analytic_signal = hilbert(new_matrix)
    # Envelope
    amplitude_envelope = np.abs(analytic_signal)
    dset_out_envelope[:,:] = amplitude_envelope
    # Carrier
    phase, freq = return_instantaneous_parameters(analytic_signal, fs)
    dset_out_carrier_phase[:,:] = phase
    dset_out_carrier_freq[:,:] = freq
    dset_out_emd[:,:] = new_matrix

    dset_out_envelope.flush()
    dset_out_carrier_phase.flush()
    dset_out_carrier_freq.flush()
    dset_out_emd.flush()
    f_in.close()
    f_out.close()


def save_instantaneous_phase_and_freq_spatial_annulus(data_file, analysis_file, create_new_file=False, fs=100):
    """
    Implements the EMD algorithm to every 'row' of a 2D dataset, using 'sum_emds'. Then performs the Hilbert Transform to obtain the envelope and saves the instantaneous frequency and phase. Saves the results in an existing file.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Opening hdf5 output file')
    f_in = h5py.File(data_file, 'r')
    dset_data = f_in['height_fields/annulus_polar']
    mask_annulus = f_in['masks/annulus']

    logger.info('Generating height averages in polar coordinates')
    matrix = dset_data[:,:]
    if create_new_file == False:
        f_out = h5py.File(analysis_file, 'r+')
    else:
        f_out = h5py.File(analysis_file, 'w')

        # Generate dataset for carrier
    dset_out_carrier_phase_spatial = f_out.create_dataset('carrier_phase_spatial_annulus', matrix.shape, chunks=True, dtype='float')
    dset_out_carrier_freq_spatial = f_out.create_dataset('carrier_freq_spatial_annulus', shape = (matrix.shape[0], matrix.shape[1]-1), chunks=True, dtype='float')

    new_matrix = np.zeros(shape=matrix.shape)
    logger.info('Performing iterations')
    for i in range(matrix.shape[0]):
      new_matrix[:,i] = sum_emds(matrix[:,i], fs)
    # Hilbert Transform
    analytic_signal = hilbert(new_matrix)
    # Carrier
    phase, freq = return_instantaneous_parameters(analytic_signal, fs)
    dset_out_carrier_phase[:,:] = phase
    dset_out_carrier_freq[:,:] = freq

    dset_out_envelope.flush()
    dset_out_carrier_phase.flush()
    dset_out_carrier_freq.flush()
    f_in.close()
    f_out.close()


 #%%




def spline_coefficients(t, z, filter_size=10):
    """
    Calculates spline coefficients of a given 1D time series.
    """
    # L = len(t)
    # N = filter_size/2
    # dt = np.mean(np.diff(t))
    # cidx = N

    # vector_of_ones = np.ones()

    # for ii in range(N, L-N):
    #     cidx = cidx + 1

    #     linear =  1

    #     A = [ unos, linear, linear**2, linear**3 ]

    #     At = A.transpose()

    #     invAtpA = np.linalg.inv( At*A )
    #     coeffs = At*x[] * invAtpA

    # return coeffs

def velocity_from_spline_coefficients():
    return None

def acceleration_from_spline_coefficients():
    return None
