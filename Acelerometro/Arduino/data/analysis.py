import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import matplotlib.gridspec as gridspec

t, x, y, z = np.loadtxt('data_baud115200_bw1600_20Hz_4vpp.txt', delimiter=',',
                        skiprows=31, unpack=True)

# Cambio de unidades a m/s^2

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

#  freqs, fft = do_fft(z, rate=1/(t[1]-t[0]))

print(np.mean(np.diff(t)))
print(np.std(np.diff(t)))
#  plt.plot(t, z, '.-')

# Plot
fig = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t, x, color='k')
ax.set_ylabel('Amplitud', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, y, color='k')
ax.grid(zorder=-100, alpha=0.4)

ax = fig.add_subplot(gs[1, :])
ax.plot(t, z, color='k')
ax.set_ylabel('Amplitud', fontsize=14)
ax.set_xlabel('Tiempo [ms]', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)

plt.savefig('img/acelerometro_xyz.png', dpi=600, bbox_inches='tight',
            transparent=True)
plt.show()
