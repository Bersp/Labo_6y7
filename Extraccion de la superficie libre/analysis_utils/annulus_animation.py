import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import h5py

hdf5_folder = '../../Mediciones_FaradayWaves/MED5 - 0716/HDF5/'
f = h5py.File(hdf5_folder+'FTP.hdf5', 'r')
annulus = f['height_fields']['annulus']

import numpy as np
import matplotlib.pyplot as plt

def surface_plot(matrix, **kwargs):
    x, y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

img = annulus[600:, 600:, 0]
#  img[np.isnan(img)] = np.nanmean(img)

(fig, ax, surf) = surface_plot(img, cmap=plt.cm.coolwarm,
                               rstride=True, antialiased=True)
fig.colorbar(surf)

ax.set_xlabel('X (cols)')
ax.set_ylabel('Y (rows)')
ax.set_zlabel('Z (values)')

plt.show()

#  st_instance0 = st_diagram[0]
#  st_instance_len = st_instance0.size
#  st_instance_dom = np.arange(st_instance_len)

#------------------------------------------------------------------------------

#  # Animation
#  def animate():
    #  fig, ax = plt.subplots()
    #  line, = plt.plot(st_diagram[0])

    #  ax.set_ylim([-0.8, 0.8])

    #  def update(frame):
        #  st_instance = st_diagram[frame]
        #  line.set_data(st_instance_dom, st_instance)

        #  s = frame//250
        #  ax.set_title(f'{s} segundos')

        #  ax.relim()
        #  ax.autoscale()

    #  ani = animation.FuncAnimation(fig, update,
                                  #  frames=range(0, st_diagram.shape[0], 1),
                                  #  blit=False, interval=60, repeat=False)

    #  plt.show()

#  # Some frames
#  def some_frames(frames):
    #  fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    #  axes = axes.flatten()

    #  for i, frame in enumerate(frames):
        #  axes[i].plot(st_diagram[frame])
        #  axes[i].set_title(f'Frame {frame}')

    #  plt.show()

#  some_frames(range(20, 28))
