import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import matplotlib.gridspec as gridspec

t, x, y, z = np.loadtxt('data_calibration/calibration1.csv', delimiter=',',
                        skiprows=200, unpack=True)

def do_fft(y: np.ndarray, rate: int) -> (np.ndarray, np.ndarray):
    fft = np.abs(np.fft.fft(y))

    freqs = np.linspace(-rate/2, rate/2, len(y), endpoint=False)

    # np.fft.fft te da primero la rama de freq negativas
    # y luego la de freq positivas. Esto lo da vuelta :).
    fft = np.fft.fftshift(fft)

    # Solo la rama de frecuencias positivas
    idx = freqs > 0
    fft = fft[idx]
    freqs = freqs[idx]

    return freqs, fft


print(np.mean(np.diff(t)))
print(np.std(np.diff(t)))
#  plt.plot(t, z, '.-')

# Calibration
res = np.load('output_calibration/calib_cobelli.npy')
O, S = res[:3], res[3:]
V = np.array([x, y, z]).T
Ax, Ay, Az = ((V - O) / S).T * 9.80665

# Plot
fig = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t, Ax, color='k')
ax.set_ylabel('Aceleración en x', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, Ay, color='k')
ax.set_ylabel('Aceleración en y', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)

ax = fig.add_subplot(gs[1, :])
ax.plot(t, Az, color='k')
ax.set_ylabel('Aceleración en z', fontsize=14)
ax.set_xlabel('Tiempo [ms]', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)

plt.show()
