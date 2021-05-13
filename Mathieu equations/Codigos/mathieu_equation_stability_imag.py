""" WIP. Contour plot de la estabilidad de la ecuación de mathieu. """

import matplotlib.pyplot as plt
import numpy as np
from numba import jit


res = 51
res_lim = (res-1)//2

def delta_coef(mu, k): return (mu-2*k)**2
DELTA_COEF_0 = np.array([delta_coef(0, i)  for i in range(-res_lim, res_lim+1)])
DELTA_COEF_1 = np.array([delta_coef(1, i)  for i in range(-res_lim, res_lim+1)])

identity = np.identity(res)

@np.vectorize
def det_delta_0(q, p):
    return np.linalg.det(
                identity
                + np.diag(q/(DELTA_COEF_0-p), 1)[:-1,:-1]
                + np.diag(q/(DELTA_COEF_0-p), -1)[1:,1:]
                )

@np.vectorize
def det_delta_1(q, p):
    return np.linalg.det(
                identity
                + np.diag(q/(DELTA_COEF_1-p), 1)[:-1,:-1]
                + np.diag(q/(DELTA_COEF_1-p), -1)[1:,1:]
                )

@np.vectorize
def mu_odd(q, p):
    return 2/np.pi * np.arcsin(np.sqrt( det_delta_0(q, p)*np.sin(np.pi/2 * np.sqrt(p))**2 ))

@np.vectorize
def mu_even(q, p):
    return 1/np.pi * np.arccos(2*det_delta_1(q, p)-1)

@np.vectorize
def mu(q, p):
    if np.real(p) % 2 == 0:
        return mu_even(q, p)
    else:
        return mu_odd(q, p)

N = 50
q = np.linspace(0.001, 11.99, N).astype('complex')
p = np.linspace(0.001, 13.99, N).astype('complex')

pp, qq = np.meshgrid(p, q)
M = mu(pp, qq)

qlims = [0.001, 12]
plims = [0.001, 14]

# Plot
fig, ax = plt.subplots(figsize=(8, 9))
im = ax.imshow(np.imag(M), cmap='gray', extent=[*qlims, *plims])
plt.show()
