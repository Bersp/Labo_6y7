import matplotlib.pyplot as plt
import numpy as np

from sympy import *
from sympy.abc import *

g = 9.80665

i = Idx('i', (1, 3))
j = Idx('j', (1, 3))
k = Idx('k', (1, N))

l = Idx('l', (1, 3))
m = Idx('m', (1, 3))
n = Idx('n', (1, 3))

O = IndexedBase('O')
V = IndexedBase('V')
S = IndexedBase('S')

Theta = [O[i] for i in range(1, 4)] + [S[i,j] for i in range(1,4)
                                              for j in range(1,4)
                                              if i <= j]

def main_plot():
    import seaborn as sns
    sns.set_style('white')

    import matplotlib.gridspec as gridspec

    N = 5
    t, x, y, z = np.loadtxt(f'data_calibration/calibration{N}.csv', delimiter=',',
                            skiprows=31, unpack=True)

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

    #  freqs, fft = do_fft(z, rate=1/(t[1]-t[0]))

    print(np.mean(np.diff(t)))
    print(np.std(np.diff(t)))
    #  plt.plot(t, z, '.-')

    # Plot
    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, x, color='k', marker='.')
    ax.set_ylabel('Aceleración en x', fontsize=14)
    ax.grid(zorder=-100, alpha=0.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, y, color='k', marker='.')
    ax.set_ylabel('Aceleración en y', fontsize=14)
    ax.grid(zorder=-100, alpha=0.4)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, z, color='k', marker='.')
    ax.set_ylabel('Aceleración en z', fontsize=14)
    ax.set_xlabel('Tiempo [ms]', fontsize=14)
    ax.grid(zorder=-100, alpha=0.4)

    plt.show()

def main_calibration():
    global J, H
    J, H = create_J_H()

    t0 = np.asarray([[1]]*9)
    alpha = 0.3

    t = t0
    for _ in range(10):
        J_eval = np.asarray(subs_theta_values(J, t), dtype=float)
        invH_eval = np.linalg.inv(np.asarray(subs_theta_values(H, t), dtype=float))

        t = t - alpha * invH_eval @ J_eval

    print(t)



    #  print(subs_theta_values(J, t0))

def subs_theta_values(M, t):
    for i, theta in enumerate(Theta):
        M = M.subs(theta, t[i,0])
    return M


def create_V_jk():
    V = np.zeros(shape=(16, 3))
    for k in range(0, 16):
        t, x, y, z = np.loadtxt(f'data_calibration/calibration{k}.csv',
                                delimiter=',', skiprows=31, unpack=True)
        g_k = np.asarray([x.mean(), y.mean(), z.mean()])

        V[k,:] = g_k
    return V

def create_J_H():
    global i,j,k, l,m,n, O,V,S, Theta

    s = Sum(
            (S[i,j] * (V[j] - O[j]))**2,
            (j, 1, 3)
            )

    # Declaro e y la simetrizo
    e = Sum(
            s - g**2,
            (i, 1, 3)
            )
    E = e**2


    Theta = [O[i] for i in range(1, 4)] + [S[i,j] for i in range(1,4)
                                                  for j in range(1,4)
                                                  if i <= j]
    J = zeros(9, 1)
    H = zeros(9, 9)
    for i, theta_i in enumerate(Theta):
        E_diff_theta_i = E.diff(theta_i)
        J[i] = (E_diff_theta_i)
        for j, theta_j in enumerate(Theta):
            E_diff_theta_i_theta_j = E_diff_theta_i.diff(theta_j)
            H[i, j] = E_diff_theta_i_theta_j

    J, H = J.doit(), H.doit()
    J = J.subs(S[2,1], S[1,2]).subs(S[3,1], S[1,3]).subs(S[3,2], S[2,3])
    H = H.subs(S[2,1], S[1,2]).subs(S[3,1], S[1,3]).subs(S[3,2], S[2,3])


    # Meto la data dentro de la matriz
    V_data = create_V_jk()

    J_subs = zeros(9, 1)
    H_subs = zeros(9)
    for v in V_data[:1]:
        J_subs += J.subs(V[1], v[0]).subs(V[2], v[1]).subs(V[3], v[2])
        H_subs += H.subs(V[1], v[0]).subs(V[2], v[1]).subs(V[3], v[2])

    return J_subs, H_subs

if __name__ == '__main__':
    #  main_plot()
    main_calibration()
