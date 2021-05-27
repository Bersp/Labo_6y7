import numpy as np
import matplotlib.pyplot as plt

t, x, y, z = np.loadtxt('data_baud115200_bw1600_20Hz_4vpp.txt', delimiter=',', skiprows=31, unpack=True)

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

freqs, fft = do_fft(z, rate=1/(t[1]-t[0]))

print(np.mean(np.diff(t)))
print(np.std(np.diff(t)))
#  plt.plot(t, z, '.-')
plt.plot(freqs, fft)
plt.show()
