import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from mpl_utils import *

# -----------------------------------------------------------------
# Funciones auxiliares


def fill_nans(data):
    """
    Rellena una los np.nan de un array de cualquier dimensión
    con los valores más cercanos distintos de nan
    """
    ind = nd.distance_transform_edt(np.isnan(data), return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]

def phase_to_height(st_diagram, L, d, p):
    """
    TODO: Docstring for phase_to_height.

    Parameters
    ----------
    st_diagram : Diagrama espacio-temporal
    L : Distancia entre el plano de referencia y la cámara
    d : Distancia entre el proyector y la cámara
    p : Longitud de onda del patrón proyectado en mm

    Returns
    -------
    El diagrama en unidades de altura (mm)
    """

    dphase = st_diagram
    return -L * dphase / (2 * np.pi * d / p - dphase)

# -----------------------------------------------------------------


def get_st_diagram(med_folder_name, error_filter=None):
    hdf5_path = f'../../Mediciones/{med_folder_name}/HDF5/ST.hdf5'
    f = h5py.File(hdf5_path, 'r')

    st_diagram = np.array(f['spatiotemporal_diagram'])
    st_error = np.array(f['spatiotemporal_diagram_error'])

    # Transformamos a alturas
    L, d, p = f.attrs['L'], f.attrs['d'], f.attrs['p']
    st_diagram = phase_to_height(st_diagram, L, d, p)

    if error_filter != None:
        st_diagram[st_error > np.mean(
            st_error)+np.std(st_error)*error_filter] = np.nan
        st_diagram = fill_nans(st_diagram)

    return st_diagram


def st(med_folder_name):
    st_diagram = get_st_diagram(med_folder_name)
    st_diagram[st_diagram > 5] = np.nan
    st_diagram[st_diagram < -8] = np.nan

    fig, ax = plt.subplots(figsize=(12, 8))

    img = ax.imshow(st_diagram)
    fig.colorbar(img)
    ax.set_xlabel('theta [pixel]', fontsize=16)
    ax.set_ylabel('time [frames]', fontsize=16)
    plt.show()


def animate_st(ylim=None):
    st_instance0 = st_diagram[0]
    x = np.linspace(0, 2*np.pi, st_instance0.size)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlabel('x [rad]', fontsize=16)
    ax.set_ylabel('"Altura"', fontsize=16)

    line, = plt.plot(x, st_instance0)
    point, = plt.plot(x[300], st_instance0[300], 'ro')

    if ylim != None:
        ax.set_ylim(*ylim)

    def update(frame):
        st_instance = st_diagram[frame]
        #  st_instance = nd.uniform_filter1d(st_instance, 10)
        line.set_data(x, st_instance)
        point.set_data(x[300], st_instance[300])

        s = frame/250
        ax.set_title(f'{s} segundos')

        if ylim == None:
            ax.relim()
            ax.autoscale()

        return line, point

    ani = animation.FuncAnimation(fig, update,
                                  frames=range(0, st_diagram.shape[0], 1),
                                  blit=True, interval=1, repeat=True)

    ani.save('animation_test.mp4', fps=250)

    plt.show()


def some_frames(med_folder_name, start_frame, interval):
    st_diagram = get_st_diagram(med_folder_name)
    frames = range(start_frame, start_frame+8*interval, interval)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, frame in enumerate(frames):
        axes[i].plot(st_diagram[frame], 'k')
        axes[i].set_title(f'Frame {frame}')
        axes[i].set_ylim(-3, 2.5)
    plt.show()


def point_through_time(med_folder_name, point):
    st_diagram = get_st_diagram(med_folder_name)

    theta_point = point * 2*np.pi / st_diagram[0].size

    data = st_diagram[:, point]
    data = nd.uniform_filter1d(data, 10)
    time = np.arange(data.size)/250  # domain in s

    fig, ax = plt.subplots()
    ax.plot(time, data, c=DARK_GRAY_NORD)
    ax.set_xlabel('Tiempo (s)', fontsize=16)
    ax.set_ylabel(f'"Altura"',
                  fontsize=16)

    ax.grid()
    plt.show()


def error_filter_analysis(med_folder_name, start=3, interval=1):
    import matplotlib
    cmap = matplotlib.cm.coolwarm
    cmap.set_bad('black')

    hdf5_folder = f'../../Mediciones/{med_folder_name}/HDF5/ST.hdf5'
    f = h5py.File(hdf5_folder, 'r')
    st_diagram = np.array(f['spatiotemporal_diagram'])
    st_error = np.abs(np.array(f['spatiotemporal_diagram_error']))

    # fig, (ax1, ax2) = plt.subplots(
    # 1, 2, figsize=(12, 8), sharex=True, sharey=True)

    # st_diagram_tmp = st_diagram.copy()
    # st_diagram_tmp[st_error > np.mean(st_error)+np.std(st_error)*3] = np.nan
    # ax1.set_title('Diagrama filtrado a $3 \sigma$', fontsize=16)
    # ax1.imshow(st_diagram_tmp, cmap=cmap)

    # ax2.set_title('Diagrama con interpolación', fontsize=16)
    # ax2.imshow(fill_nans(st_diagram), cmap=cmap)

    # ax1.set_xlim(600, 1000)
    # ax1.set_ylim(1300, 1700)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # plt.savefig('/home/bersp/error_interp.pdf', dpi=300, bbox_inches='tight',
    # transparent=True)
    # plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    cmap.set_bad('black')
    axes[0].set_title('Completa')
    axes[0].imshow(st_diagram, cmap=cmap)
    for i in range(start, start+5, interval):
        st_diagram_tmp = st_diagram.copy()
        st_diagram_tmp[st_error > np.mean(
            st_error)+np.std(st_error)*i] = np.nan

        axes[i-start +
             1].set_title('NaN si $ err > \\overline{err} +'+str(i)+'\sigma $')
        axes[i-start+1].imshow(st_diagram_tmp, cmap=cmap)

    axes[0].set_xlim(600, 1000)
    axes[0].set_ylim(1300, 1700)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # plt.savefig('/home/bersp/error_analysis.pdf', dpi=300,
                # bbox_inches='tight', transparent=True)
    plt.show()


def main():
    # med_folder_name = 'MED63 - Bajada en voltaje - NOTE - 1104/'
    # med_folder_name = 'MED36 - Subida en voltaje - 0902/'
    med_folder_name = 'MED69 - Diversion - 1104'
    st_diagram = get_st_diagram(med_folder_name)
    st_diagram = np.diff(st_diagram, axis=0)
    plt.imshow(st_diagram, clim=(-2, 2))
    plt.colorbar()
    plt.show()
    
    
    # error_filter_analysis(med_folder_name, start=3)
    # plt.colorbar()
    
    # st(med_folder_name)
    # animate_st(ylim=(-5, 5))
    # some_frames(med_folder_name, start_frame=20, interval=30)
    # point_through_time(med_folder_name, point=300)
    
if __name__ == '__main__':
    main()
