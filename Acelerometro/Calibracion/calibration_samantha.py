import numpy as np
from scipy.linalg import pinv

def get_V():
    N_meds = 16
    V = np.zeros(shape=(3, N_meds))
    for k in range(0, N_meds):
        t, x, y, z = np.loadtxt(f'data_calibration/calibration{k}.csv',
                                delimiter=',', skiprows=31, unpack=True)
        v_k = np.asarray([x.mean(), y.mean(), z.mean()])

        V[:,k] = v_k
    return V

dataX, dataY, dataZ = get_V()

N = dataX.size

zeta = np.zeros(shape=(N, 6))

zeta[:, 0] = dataX**2
zeta[:, 1] = dataY**2
zeta[:, 2] = dataZ**2
zeta[:, 3] = -2*dataX
zeta[:, 4] = -2*dataY
zeta[:, 5] = -2*dataZ

zeta = np.matrix(zeta)

gval = 9.81
g = (gval**2)*np.ones(shape=(N,1))

xi = pinv(zeta.T*zeta)*zeta.T*g

xi, = np.array(xi.T)

C = 1 + xi[3]**2/xi[0] + xi[4]**2/xi[1] + xi[5]**2/xi[2]
S = np.sqrt(C/xi[0:3]);
O = xi[3:6]/xi[0:3];

S = abs(S);

print(f'{S = }\n')
print(f'{O = }')

