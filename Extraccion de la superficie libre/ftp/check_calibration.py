import numpy as np
import matplotlib.pyplot as plt

filename = '../../Acelerometro/Calibracion/output_calibration/calib_cobelli.npy'

o = np.load(filename)
O, S = o[:3], o[3:]

t, x, y, z = np.loadtxt('../../Mediciones_FaradayWaves/MED8 - 0730/acelerometer/acceleration.csv',
                     delimiter=',', unpack=True)


print(O, S)
med = np.array([x, y, z]).T
Ax, Ay, Az = ((med - O) / S).T

# one = ((med - O)**2 / S**2).sum(1)
# print(one)

plt.plot(t, Ay)
plt.show()
