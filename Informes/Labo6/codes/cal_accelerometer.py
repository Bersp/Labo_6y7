import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes

from mpl_utils import *

cal_folder = '../../../Acelerometro/Calibracion/'
calib = np.load(cal_folder+'output_calibration/calib_cobelli.npy')
O, S = calib[:3], calib[3:]

t, x, y, z = np.loadtxt(cal_folder+'data_calibration/test_calibration_onoff_20hz_10vpp.csv',
                        delimiter=',', skiprows=200, unpack=True)
t /= 1000 # time to s

# Test med
V = np.array([x, y, z]).T
Ax, Ay, Az = ((V - O) / S).T * 9.81

fig = plt.figure(figsize=(13,8))
fig.subplots_adjust(wspace=0.3)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t, Ax, color=DARK_GRAY_NORD)
ax.set_ylabel('Aceleración en $x$ $[m/s^2]$', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)
ax.set_xlim(0, 21)

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, Ay, color=DARK_GRAY_NORD)
ax.set_ylabel('Aceleración en $y$ $[m/s^2]$', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)
ax.set_xlim(0, 21)

ax = fig.add_subplot(gs[1, :])
ax.plot(t, Az, color=DARK_GRAY_NORD)
ax.set_ylabel('Aceleración en $z$ $[m/s^2]$', fontsize=14)
ax.set_xlabel('Tiempo [s]', fontsize=14)
ax.grid(zorder=-100, alpha=0.4)
ax.set_xlim(0, 21)

axins = inset_axes(ax, 2.9, 1.35, loc=1)
axins.plot(t, Az, color=DARK_GRAY_NORD)

axins.set_xlim(5, 6)
axins.set_ylim(4, 16.5)

axins.set_xticklabels('')
axins.set_yticklabels('')

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=BLACK_NORD, zorder=10)

plt.tight_layout()
plt.savefig(f'../figures/calib_accelerometer.pdf', transparent=True)
plt.show()
