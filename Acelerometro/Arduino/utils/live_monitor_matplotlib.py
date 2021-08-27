import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import numpy as np

def get_data():
    while True:
        z = []
        while len(z) != 500:
            try:
                t, x, y, z = np.loadtxt(filename, delimiter=',', unpack=True)
            except:
                t = [None]

        if t[0]:
            yield t, z

def update(data):
    if data:
        t, z = data
        line.set_data(t, z)

        max_t = np.max(t)
        ax.set_xlim(max_t-100, max_t)

filename = 'data.csv'
t, x, y, z = [np.nan]*4

fig, ax = plt.subplots()
line, = plt.plot(t, z, '.-')
ax.set_ylim(-512, 512)
#  ax.set_xlim(0, 2*np.pi)

ani = animation.FuncAnimation(fig, update, frames=get_data(),
                              blit=False, interval=100, repeat=False)

plt.show()
