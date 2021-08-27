import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from mpl_utils import *
from scipy import ndimage as nd
from skimage.restoration import unwrap_phase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes

from unwrap import unwrap

def fill_nans(data):
    """
    Rellena una los np.nan de un array de cualquier dimensión
    con los valores más cercanos distintos de nan
    """
    ind = nd.distance_transform_edt(np.isnan(data), return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_st_diagram(med_folder_name):
    hdf5_path = f'../../../Mediciones_FaradayWaves/{med_folder_name}/HDF5/ST.hdf5'
    f = h5py.File(hdf5_path, 'r')

    st_diagram = np.array(f['spatiotemporal_diagram'])

    gap = st_diagram[:, 100] - np.unwrap(st_diagram[:, 100])

    gap = np.expand_dims(gap, 1)
    st_diagram = st_diagram - gap

    st_diagram = unwrap(st_diagram)

    st_diagram -= np.expand_dims(st_diagram.mean(1), 1)

    st_diagram[(st_diagram > 5) + (st_diagram < -5)] = np.nan
    st_diagram = fill_nans(st_diagram)

    st_diagram *= -1

    #  st_diagram = np.remainder(st_diagram, 2*np.pi)
    #  st_diagram[st_diagram < 1.2] = np.nan
    #  st_diagram = fill_nans(st_diagram)
    #  st_diagram = np.pi - st_diagram

    #  st_diagram = (st_diagram.T - st_diagram.mean(1)).T
    #  mask = (st_diagram > 5) + (st_diagram < -5)
    #  st_diagram[mask] = np.nan
    #  st_diagram = fill_nans(st_diagram)
    #  st_diagram *= -1

    return st_diagram

def st(med_folder_name):
    st_diagram = get_st_diagram(med_folder_name)
    extent = (0, 2*np.pi, 0, st_diagram.shape[1]/250)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = truncate_colormap('twilight_shifted', 0.1, 0.90)

    img = ax.imshow(st_diagram,
                    extent=extent,
                    aspect=0.5,
                    cmap=cmap
                    )
    fig.colorbar(img, ax=ax)

    ax.set_xlabel('theta [rad]', fontsize=20)
    ax.set_ylabel('time [s]', fontsize=20)

    axins = zoomed_inset_axes(ax, 4.5, loc=2)
    axins.imshow(st_diagram,
                 extent=extent,
                 aspect=0.5,
                 cmap=cmap
                 )

    axins.set_xlim(5.5, 6)
    axins.set_ylim(2, 3)

    axins.set_xticklabels('')
    axins.set_yticklabels('')

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec=BLACK_NORD)


    if '5' in med_folder_name:
        state='Baja aceleración'
        tag_ax(ax, state, pos=(0.80, 0.95))

    elif '11' in med_folder_name:
        state='Alta aceleración'
        tag_ax(ax, state, pos=(0.80, 0.95))


    plt.tight_layout()
    plt.savefig(f'../figures/st_{med_folder_name.split(" ")[0]}.pdf',
                transparent=True, bbox_inches='tight')
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

def point_through_time(point):
    fig, axes = plt.subplots(2, sharex=True)

    for ax, med_folder_name in zip(axes, ['MED5 - 0716', 'MED11 - 0730']):

        st_diagram = get_st_diagram(med_folder_name)

        theta_point = point * 2*np.pi/ st_diagram[0].size

        data = st_diagram[:, point]
        data = uniform_filter1d(data, 10)
        time = np.arange(data.size)/250 # domain in s

        ax.plot(time, data, c=DARK_GRAY_NORD)

        ax.set_ylabel(r'$\Delta \phi$', fontsize=16)

        med = med_folder_name.split(' ')[0]
        tag_ax(ax, text=med)

        ax.grid()

    ax.set_xlabel('Tiempo (s)', fontsize=16)

    plt.tight_layout()
    plt.savefig('../figures/med5_med11_thetatt.png', transparent=True)

    plt.show()

def animate_st():
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10), sharex=True)

    p = 300
    theta = p * 2*np.pi/3000

    ax1.axhline(0, c=BLACK_NORD, alpha=0.5, zorder=-1)
    ax2.axhline(0, c=BLACK_NORD, alpha=0.5, zorder=-1)

    # -------------------------------------------------------------------------
    # MED 5
    st_diagram_m5 = get_st_diagram('MED5 - 0716')

    st_instance0 = st_diagram_m5[0]
    x = np.linspace(0, 2*np.pi, st_instance0.size)
    #  x = range(st_instance0.size)

    ax1.set_ylabel(r'$\Delta \phi$', fontsize=16)

    line_m5, = ax1.plot(x, st_instance0, c=DARK_GRAY_NORD)

    #  point_m5, = ax1.plot(x[p], st_instance0[p], marker='o', c=RED_NORD)

    ax1.set_ylim(-0.75, 1)
    tag_ax(ax1, 'Baja aceleración', 'bottomright')

    # -------------------------------------------------------------------------
    # MED 11
    st_diagram_m11 = get_st_diagram('MED11 - 0730')

    # Cortes razonables para mejorar la visualización
    st_diagram_m11[:, 2932:3936] = np.nan

    st_instance0 = st_diagram_m11[0]

    ax2.set_xlabel(r'$\theta$ [rad]', fontsize=16)
    ax2.set_ylabel(r'$\Delta \phi$', fontsize=16)

    line_m11, = ax2.plot(x, st_instance0, c=DARK_GRAY_NORD)

    #  point_m11, = ax2.plot(x[p], st_instance0[p], marker='o', c=RED_NORD)

    ax2.set_ylim(-2.5, 4.5)
    tag_ax(ax2, 'Alta aceleración', 'bottomright')

    # -------------------------------------------------------------------------
    # Animation

    def update(frame):
        st_instance = st_diagram_m5[frame]
        line_m5.set_data(x, st_instance)
        #  point_m5.set_data(x[p], st_instance[p])

        st_instance = st_diagram_m11[frame]
        line_m11.set_data(x, st_instance)
        #  point_m11.set_data(x[p], st_instance[p])

        s = frame/250
        #  ax.set_title(f'{s} segundos')

        return line_m5, line_m11

    ani = animation.FuncAnimation(fig, update,
                                  frames=range(0, st_diagram_m11.shape[0], 1),
                                  blit=True, interval=1, repeat=True)

    plt.tight_layout()
    #  ani.save('../media/animation_med5_med11_st.mp4', fps=250, dpi=200)

    plt.show()

def plot_one_line(med_folder_name, n_column=10):
    st_diagram = get_st_diagram(med_folder_name)

    fig, ax = plt.subplots(figsize=(12, 8))
    gap = st_diagram[:, n_column] - np.unwrap(st_diagram[:, n_column])

    st_diagram = (st_diagram.T - gap).T

    ax.imshow(unwrap(st_diagram))
    # plt.colorbar()

    plt.show()


if __name__ == "__main__":
    #  point_through_time(point=100)
     # animate_st()
    meds = ['MED5 - 0716', 'MED11 - 0730', 'MED12 - 0730']
    st(meds[2])
