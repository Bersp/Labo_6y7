import numpy as np
from scipy.optimize import minimize

def get_V():
    N_meds = 16
    V = np.zeros(shape=(3, N_meds))
    for k in range(0, N_meds):
        t, x, y, z = np.loadtxt(f'data_calibration/calibration{k}.csv',
                                delimiter=',', skiprows=31, unpack=True)
        v_k = np.asarray([x.mean(), y.mean(), z.mean()])

        V[:,k] = v_k
    return V

V = get_V()

def E(x0):
    Ox, Oy, Oz, Sxx, Syy, Szz, Sxy, Sxz, Syz = x0
    S = np.matrix([
            [Sxx, Sxy, Sxz],
            [Sxy, Syy, Syz],
            [Sxz, Syz, Szz]
            ])

    O = np.matrix([Ox, Oy, Oz]).T

    g = 9.81
    e_k = np.sum(np.array(S*(V-O))**2, 0) - g**2

    return np.sum(e_k**2) / e_k.size


O = np.mean(V, 1)
              #O  #Identity
x0 = np.array([*O, 1, 1, 1, 0, 0, 0])
print(E(x0))

res = minimize(E, x0, method='Powell', options={'maxiter': 1_000})

Ox, Oy, Oz, Sxx, Syy, Szz, Sxy, Sxz, Syz = res['x']
print(f"{res['success'] = }")

S = np.matrix([
        [Sxx, Sxy, Sxz],
        [Sxy, Syy, Syz],
        [Sxz, Syz, Szz]
        ])
O = np.matrix([Ox, Oy, Oz]).T

print(E((Ox, Oy, Oz, Sxx, Syy, Szz, Sxy, Sxz, Syz)))

print(f'S\n{S}\n')
print(f'O\n{O}\n')
