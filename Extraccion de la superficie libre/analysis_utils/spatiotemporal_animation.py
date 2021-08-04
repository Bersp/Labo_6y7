import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import h5py

hdf5_folder = '../../Mediciones_FaradayWaves/MED5 - 0716/HDF5/'
f = h5py.File(hdf5_folder+'ST.hdf5', 'r')
st_diagram = f['spatiotemporal_diagram']

st_instance0 = st_diagram[0]
st_instance_len = st_instance0.size
st_instance_dom = np.arange(st_instance_len)


# Animation
def animate():
    fig, ax = plt.subplots()
    line, = plt.plot(st_diagram[0])

    ax.set_ylim([-0.8, 0.8])

    def update(frame):
        st_instance = st_diagram[frame]
        line.set_data(st_instance_dom, st_instance)

        s = frame//250
        ax.set_title(f'{s} segundos')

        ax.relim()
        ax.autoscale()

    ani = animation.FuncAnimation(fig, update,
                                  frames=range(0, st_diagram.shape[0], 1),
                                  blit=False, interval=60, repeat=False)

    plt.show()

# Some frames
def some_frames(frames):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, frame in enumerate(frames):
        axes[i].plot(st_diagram[frame])
        axes[i].set_title(f'Frame {frame}')

    plt.show()

some_frames(range(20, 28))
