import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

fig, ax = plt.subplots()

t, x, y, z = np.loadtxt('data.csv', delimiter=',', unpack=True)
line, = plt.plot(t, z)

def get_data():
    while True:
        try:
            t, x, y, z = np.loadtxt('data.csv', delimiter=',', unpack=True)
        except:  # Cuando el archivo está vacío numpy lanza un warning que
            pass # intenta detener la ejetución. TODO: Hacer esto mejor.
        yield t, z

def update(data):
    t, z = data
    line.set_data(t, z)

    ax.relim()
    ax.autoscale()

ani = animation.FuncAnimation(fig, update, frames=get_data(),
                              blit=False, interval=1, repeat=False)

plt.show()
