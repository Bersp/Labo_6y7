from raw_data_utils import create_raw_hdf5
from ftp import FTP
from spatiotemporal_diagram import create_st_hdf5

med_folder = '../../Mediciones_FaradayWaves/MED12 - 0730/'
hdf5_folder = med_folder+'HDF5/'

# RAW
create_raw_hdf5(med_folder)

# FTP
FTP(hdf5_folder).export()

# ST
create_st_hdf5(hdf5_folder)

# Plot ST diagram
import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File(hdf5_folder+'ST.hdf5', 'r')
st_diagram = np.array(f['spatiotemporal_diagram'])
st_diagram = (st_diagram.T - st_diagram.mean(1)).T

st_diagram[st_diagram > 5] = np.nan
st_diagram[st_diagram < -8] = np.nan

plt.imshow(st_diagram)
plt.xlabel('theta')
plt.ylabel('Time (frame)')
plt.colorbar()
plt.show()
