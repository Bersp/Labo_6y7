import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

import matplotlib.gridspec as gridspec

def get_V():
    from_med = 0
    to_med = 9
    V = np.zeros(shape=(3, to_med-from_med))
    for k in range(to_med-from_med):
        t, x, y, z = np.loadtxt(f'data_calibration/calibration{k+from_med}.csv',
                                delimiter=',', skiprows=200, unpack=True)
        v_k = np.asarray([x.mean(), y.mean(), z.mean()])

        V[:,k] = v_k
    return V

V = get_V()

Ax, Ay, Az = get_V()

N = Ax.size

M = np.zeros((N, 6))

M[:, 0] = Ax**2
M[:, 1] = Ay**2
M[:, 2] = Az**2

M[:, 3] = -2*Ax
M[:, 4] = -2*Ay
M[:, 5] = -2*Az

alpha = pinv(M.T @ M) @ M.T @ np.ones(N)

# Calculo O
O = np.zeros(3)
O = alpha[3:]/alpha[:3]

# Caculo S
C = 1 + (alpha[3:]**2/alpha[:3]).sum()
S = np.sqrt(C/alpha[:3])

print(f'{O = }')
print(f'{S = }')

# --------------------------------------------------------------------
# Test
np.save('output_calibration/calib_cobelli.npy', np.concatenate([O, S]))

t, x, y, z = np.loadtxt(f'data_calibration/test_calibration_onoff_20hz_10vpp.csv',
                        delimiter=',', skiprows=200, unpack=True)
# Test med
V = np.array([x, y, z]).T
Ax, Ay, Az = ((V - O) / S).T

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
