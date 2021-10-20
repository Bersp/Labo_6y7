import h5py


f = h5py.File('FTP.hdf5', 'r')

print(f.attrs['L'])
