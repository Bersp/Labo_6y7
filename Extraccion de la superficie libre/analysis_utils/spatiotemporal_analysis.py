import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from mpl_utils import *

def get_st_diagram(med_folder_name):
    hdf5_path = f'../../Mediciones_FaradayWaves/{med_folder_name}/HDF5/ST.hdf5'
    f = h5py.File(hdf5_path, 'r')

    st_diagram = np.array(f['spatiotemporal_diagram'])
    st_diagram = (st_diagram.T - st_diagram.mean(1)).T

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
        #  st_instance = uniform_filter1d(st_instance, 10)
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

    theta_point = point * 2*np.pi/ st_diagram[0].size

    data = st_diagram[:, point]
    data = uniform_filter1d(data, 10)
    time = np.arange(data.size)/250 # domain in s


    fig, ax = plt.subplots()
    ax.plot(time, data, c=DARK_GRAY_NORD)
    ax.set_xlabel('Tiempo (s)', fontsize=16)
    ax.set_ylabel(f'"Altura"',
                  fontsize=16)

    ax.grid()
    plt.show()

if __name__ == "__main__":
    med_folder_name = 'MED11 - 0730'
    st(med_folder_name)
    #  animate_st(ylim=(-5, 5))
    #  some_frames(med_folder_name, start_frame=20, interval=30)
    #  point_through_time(med_folder_name, point=300)
