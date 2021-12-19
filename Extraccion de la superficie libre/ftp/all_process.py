import logging
from operator import itemgetter
import os
import re

from spatiotemporal_diagram import create_st_hdf5

# Logger config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')


def all_process(med_folder):
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


def all_process_parallel(med_folder, n_cores=16):
    hdf5_folder = med_folder+'HDF5/'

    # RAW
    os.system(
        f'mpiexec -n {n_cores} --oversubscribe python raw_data_utils_parallel.py "{med_folder}"')

    # FTP
    os.system(
        f'mpiexec -n {n_cores} --oversubscribe python ftp_parallel.py "{med_folder}"')

    # ST
    create_st_hdf5(hdf5_folder)


def multiple_all_process_parallel(meds_folder, med_start, med_end):
    """
    TODO: Docstring for multiple_all_process_parallel.

    Parameters
    ----------
    med_start : TODO
    med_end : TODO
    """

    meds_to_process = [
        p for p in sorted(os.listdir(meds_folder)) if 'MED' in p and
        med_start <= int(re.findall(r'MED(\d+) - ', p)[0]) <= med_end
    ]

    for med_folder in meds_to_process:
        med_folder = meds_folder + med_folder + '/'

        logging.info(f'ALL_PROCESS: COMENZANDO MEDICIÓN PARA {med_folder}')
        all_process_parallel(med_folder)
        os.remove(med_folder+'HDF5/RAW.hdf5')
        os.remove(med_folder+'HDF5/FTP.hdf5')


def main():
    meds_folder = '../../Mediciones/'
    # multiple_all_process_parallel(meds_folder, 45, 62)

    # med_folder = meds_folder + 'MED40 - Oscilones a full - 0902/'
    med_folder = meds_folder + 'MED64 - Bajada en voltaje - 1104/'
    all_process_parallel(med_folder)

    # # Plot ST diagram
    # import h5py
    # import matplotlib.pyplot as plt
    # import numpy as np

    # f = h5py.File(med_folder+'HDF5/ST.hdf5', 'r')

    # st_diagram = np.array(f['spatiotemporal_diagram'])

    # plt.imshow(st_diagram, cmap='coolwarm')
    # plt.colorbar()
    # # plt.plot(st_diagram[10,:], '.-')
    # plt.xlabel('theta')
    # plt.ylabel('Time (frame)')
    # plt.show()


if __name__ == "__main__":
    main()
