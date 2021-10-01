import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

Vpp = np.arange(1, 19)
App = np.zeros(Vpp.size)
for i, vpp in enumerate(Vpp):
    t, x, y, z = np.loadtxt(f'volt_to_acc/vpp{vpp}_20hz.csv', delimiter=',', unpack=True)

    # Calibration
    res = np.load('output_calibration/calib_cobelli.npy')
    O, S = res[:3], res[3:]
    V = np.array([x, y, z]).T
    Ax, Ay, Az = ((V - O) / S).T * 9.80665

    peaks, _ = find_peaks(z, width=4)

    App[i] = np.mean(Az[peaks])

plt.plot(Vpp, App, 'o')
# plt.plot(t[peaks], Az[peaks], 'o')
# plt.title(App)
plt.show()
