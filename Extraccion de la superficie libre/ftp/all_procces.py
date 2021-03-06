from raw_data_utils import create_raw_hdf5
from ftp import FTP
from spatiotemporal_diagram import create_st_hdf5

med_folder = '../../Mediciones/MED14 - Transicion - 0826/'
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

plt.imshow(st_diagram)
plt.xlabel('theta')
plt.ylabel('Time (frame)')
plt.colorbar()
plt.show()
