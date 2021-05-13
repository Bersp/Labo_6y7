""" Gráfico de las zonas de estabilidad de la ecuación de Mathieu. """

from numba import jit
import matplotlib.pyplot as plt
import numpy as np

res = 11
res_lim = (res-1)//2

# Término (mu - 2*k)^2 de Delta(mu). Se calcula antes por optimización
delta_k_term = np.array([(2*k)**2 for k in range(-res_lim, res_lim+1)])

# Se calcula la identidad de res*res antes por optimización
identity = np.identity(res)


@jit # just in time compilation
def mu(q, p):

    # Contruyo la matriz Delta(mu=0, q, p)
    coef = q/(delta_k_term-p)
    delta_matrix = identity + np.diag(coef[:-1], 1) + np.diag(coef[1:], -1)

    # Calculo el determinante
    det = np.linalg.det(delta_matrix)

    # Si mu(q, p) es complejo retorno True; si es real, False
    if det < 0 or det*np.sin(np.pi/2 * np.sqrt(p))**2 > 1:
        return True
    else:
        return False

# Vectorizo la función con resultados del tipo bool
mu = np.vectorize(mu, otypes=[bool])

# Límites de q y p, no comienza de 0 para evitar los resultados pares
# que tienen una solución distinta (NOTE: consultar)
qlims = [0.001, 12]
plims = [0.001, 14]

q = np.linspace(*qlims, 1_000)
p = np.linspace(*plims, 1_000)
qq, pp = np.meshgrid(q, p)

# Calculo mu para todos los pares (q, p)
M = mu(qq, pp)

# Plot
fig, ax = plt.subplots(figsize=(8, 9))
ax.imshow(M[::-1], cmap='Blues', extent=[*qlims, *plims])

ax.set_xlabel(r'$q(k, f\,; \Omega, h)$', fontsize=16)
ax.set_ylabel(r'$p(k\,; \Omega, h, \rho, \sigma)$', fontsize=16)

plt.show()
