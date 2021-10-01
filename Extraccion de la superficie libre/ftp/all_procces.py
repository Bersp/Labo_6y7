import os
from spatiotemporal_diagram import create_st_hdf5

def main():
    med_folder = '../../Mediciones/MED13 - Transicion - 0826/'
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
    
    plt.imshow(st_diagram, cmap='coolwarm')
    plt.xlabel('theta')
    plt.ylabel('Time (frame)')
    plt.colorbar()
    plt.show()

def main_parallel():
    med_folder = '../../Mediciones/MED43 - Mod de fase - 0909/'
    
    # RAW
    os.system(f'mpiexec -n 16 --oversubscribe python raw_data_utils_parallel.py "{med_folder}"')
    
    # FTP
    os.system(f'mpiexec -n 16 --oversubscribe python ftp_parallel.py "{med_folder}"')
    
    # ST
    hdf5_folder = med_folder+'HDF5/'
    create_st_hdf5(hdf5_folder)
    
    # Plot ST diagram
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    
    f = h5py.File(hdf5_folder+'ST.hdf5', 'r')
    
    st_diagram = np.array(f['spatiotemporal_diagram'])
    
    plt.imshow(st_diagram, cmap='coolwarm')
    plt.xlabel('theta')
    plt.ylabel('Time (frame)')
    plt.colorbar()
    plt.show()

def main_modificated():
    med_folder = '../../Mediciones/Test/'
    hdf5_folder = med_folder+'HDF5/'
    
    # RAW
    # os.system(f'mpiexec -n 16 --oversubscribe python raw_data_utils_parallel.py "{med_folder}"')
    
    # FTP
    # os.system(f'mpiexec -n 16 --oversubscribe python ftp_parallel.py "{med_folder}"')
    
    # ST
    # create_st_hdf5(hdf5_folder)
    
    # Plot ST diagram
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    
    f = h5py.File(hdf5_folder+'ST.hdf5', 'r')
    
    st_diagram = np.array(f['spatiotemporal_diagram'])
    
    # plt.imshow(st_diagram, cmap='coolwarm', clim=(-2, 2))
    # plt.colorbar()
    plt.plot(st_diagram[10,:], '.-')
    plt.xlabel('theta')
    plt.ylabel('Time (frame)')
    plt.show()


if __name__ == "__main__":
    main_modificated()
