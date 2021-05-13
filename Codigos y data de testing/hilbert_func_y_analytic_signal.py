import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, square

def do_fft(y: np.ndarray, rate: int) -> (np.ndarray, np.ndarray):
    fft = np.real(np.fft.fft(y))

    freqs = np.linspace(-rate/2, rate/2, len(y), endpoint=False)

    # np.fft.fft te da primero la rama de freq negativas
    # y luego la de freq positivas. Esto lo da vuelta :).
    fft = np.fft.fftshift(fft)

    # Solo la rama de frecuencias positivas
    #  idx = freqs > 0
    #  fft = fft[idx]
    #  freqs = freqs[idx]

    return freqs, fft

dt = 0.0005
x = np.arange(0, 10*np.pi, dt)

cuadrado = (square(x)+1)/2

def f(x): return x
cuadrado[0:len(x)//2] = f(x)[0:len(x)//2]

y = np.sin(10*x)*np.sin(x)*cuadrado
y_err = y + np.random.normal(0, 0.1, len(x))

freqs, fft = do_fft(y_err, rate= 2*np.pi/dt)

hilbert_fft = np.fft.ifft(np.sign(y_err)*np.fft.fft(y_err))

analytic_sig = hilbert(y_err)
hilbert_sp = np.imag(analytic_sig)


plt.plot(x, y_err)
plt.plot(x, np.abs(analytic_sig))
plt.plot(x, np.abs(hilbert(y)))
plt.show()
