import os
import re

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.fft as fft
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns

from mpl_utils import *
from spatiotemporal_analysis import get_st_diagram
# sns.set_palette(sns.color_palette("rocket", 5))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


cmap = truncate_colormap('twilight_shifted', 0.1, 0.90)


def get_zeros(signal):
    a = np.isclose(signal, 0, atol=5 * np.mean(np.abs(np.diff(signal))))
    x = np.arange(signal.size)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
    ax1.plot(x, signal, '.')
    ax1.plot(x[a], signal[a], 'ro')

    b = -np.gradient(x[a])
    # b = uniform_filter1d(b, size=5)
    ax2.plot(x[a], b, '.-k')
    ax2.axhline(b.mean(), color='k')
    plt.show()
    pass


def get_envelope(signal, plot=False):
    dom_signal = np.arange(signal.size)

    max_0dist = np.max(
        np.diff(
            np.where(
                np.isclose(signal,
                           0,
                           atol=1.1 * np.abs(np.min(np.diff(signal)))))))

    if max_0dist == 1:
        max_0dist += 1

    signal = uniform_filter1d(signal, size=signal.size // 1000)
    peaks, _ = find_peaks(signal, distance=max_0dist * 0.8, height=0)

    xp, yp = peaks, signal[peaks]

    yp_ext = np.concatenate((yp, yp[:1]))
    xp_ext = np.concatenate((xp, xp[:1] + signal.size))
    envelope = CubicSpline(xp_ext, yp_ext, bc_type='periodic')

    x = dom_signal
    y = envelope(x)

    if plot:
        fig, ax = plt.subplots(1, figsize=(14, 8), sharex=True, sharey=True)

        ax.plot(dom_signal, signal, '--k', zorder=0, label='Señal original')
        ax.plot(xp, yp, 'ok', label='Máximos')
        ax.plot(x, y, '-', c=RED_NORD, label='Envolvente', lw=2)


        ax.set_ylabel('Altura [u. a.]', fontsize=18)
        ax.set_xlabel('Ángulo [rad]', fontsize=18)
        ax.set_xticks(np.linspace(0, 3000, 5, endpoint=True))
        ax.set_xticklabels(['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$', '$2\pi$'], fontsize=16)
        ax.tick_params(axis='y', which='major', labelsize=18)

        ax.grid()
        ax.legend(fontsize=14, ncol=3, framealpha=0.2)

        plt.savefig(
            f'../../Informes/Labo7_presentacion/figs/env_1sample.pdf',
            bbox_inches='tight',
            transparent=True)

        plt.show()

    return y


def get_st_spatial_envelope(st_diagram):
    """
    TODO: Docstring for get_st_envelope.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    """
    st_envelope = np.zeros(shape=st_diagram.shape)

    for i, line in enumerate(st_diagram):
        env_line = get_envelope(line)
        st_envelope[i, :] = env_line

    return st_envelope


def get_st_envelope(st_diagram):
    """
    TODO: Docstring for get_st_envelope.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    """
    st_spatial_envelope = get_st_spatial_envelope(st_diagram)
    st_envelope = get_st_spatial_envelope(st_spatial_envelope.T).T

    return st_envelope


def plot_st_envelope(st_diagram):
    def plot_aux(ax, diag, title):
        img = ax.imshow(diag, cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=18, pad=15)
        # fig.colorbar(img, ax=ax1)

    st_spatial_envelope = get_st_spatial_envelope(st_diagram)
    st_envelope = get_st_envelope(st_diagram)

    fig, (ax1, ax2, ax3) = plt.subplots(1,
                                        3,
                                        figsize=(16, 8),
                                        sharex=True,
                                        sharey=True)

    plot_aux(ax1, st_diagram, 'Diagrama original')

    plot_aux(ax2, st_spatial_envelope, 'Envolvente espacial')

    plot_aux(ax3, st_envelope, 'Envolvente espacio-temporal')

    ax1.set_xticks([])
    ax1.set_yticks([])
    # bbox_inches='tight', transparent=True)

    plt.savefig(
        f'../../Informes/Labo7_presentacion/figs/st_envelopes.pdf',
        bbox_inches='tight',
        transparent=True)

    plt.show()


def get_st_left_right(st_diagram, plot=False):
    """
    TODO: Docstring for get_st_left_right.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    """
    st_envelope = get_st_envelope(st_diagram)
    # st_envelope = get_st_spatial_envelope(st_diagram)

    N, M = st_envelope.shape

    st_fft = fft.fft2(st_envelope)
    st_fft_shifted = fft.fftshift(st_fft)

    st_fft_left_filter = np.zeros_like(st_fft_shifted)
    st_fft_right_filter = np.zeros_like(st_fft_shifted)
    st_fft_left_filter[N // 2:, M // 2:] = 1
    st_fft_right_filter[N // 2:, :M // 2] = 1

    st_fft_left = fft.ifftshift(st_fft_left_filter * st_fft_shifted)
    st_fft_right = fft.ifftshift(st_fft_right_filter * st_fft_shifted)
    st_left, st_right = fft.ifft2(st_fft_left), fft.ifft2(st_fft_right)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1,
                                            3,
                                            figsize=(12, 8),
                                            sharex=True,
                                            sharey=True)

        vmax = np.max(st_envelope)  # * .5
        vmin = np.min(st_envelope)
        img = ax1.imshow(st_envelope, cmap=cmap,
                         origin='lower')  #, vmin=vmin, vmax=vmax)
        ax1.set_title('Completa', fontsize=16, pad=10)
        # fig.colorbar(img, ax=ax1)

        # vmax = np.max(np.real(st_left)) * .5
        img = ax2.imshow(np.real(st_left), cmap=cmap,
                         origin='lower')  #, vmin=vmin, vmax=vmax)
        ax2.set_title('Izquierda', fontsize=16, pad=10)
        # fig.colorbar(img, ax=ax2)

        # vmax = np.max(np.real(st_right)) * .5
        ax3.imshow(np.real(st_right), cmap=cmap,
                   origin='lower')  #, vmin=vmin, vmax=vmax)
        ax3.set_title('Derecha', fontsize=16, pad=10)
        # fig.colorbar(img, ax=ax3)

        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.savefig(
            f'../../Informes/Labo7_presentacion/figs/st_left_right_taylor_couette.pdf',
            bbox_inches='tight',
            transparent=True)

        plt.show()

    return np.real(st_left), np.real(st_right)


def dispersion_relation(st_diagram):
    '''
    TODO: Docstring for dispersion_relation.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    '''

    M, N = st_diagram.shape
    radius = 10.5 / 100

    st_fft = fft.fft2(st_diagram)
    st_fft_shifted = fft.fftshift(st_fft)

    # Freqs
    delta_t = 1 / 250
    delta_x = radius / 512  #2 * np.pi / N  * radius # que este N esté en el medio es muy raro
    w_ny = 1 / (2 * delta_t) * 2 * np.pi
    k_ny = 1 / (2 * delta_x) * 2 * np.pi  # == N/2r

    # st_fft_filter = np.zeros_like(st_fft_shifted)
    # st_fft_filter[M // 2:, N // 2:] = 1

    # Relación de dispersión teórica
    g = 9.81  # gravedad en el tierra
    h = 1 / 100  # profundidad (anda mejor con 1/80)
    gamma = 72.75e-3  # tensión superficial
    rho = 0.998e3  # densidad del agua

    def w_gam(k):
        return np.sqrt(g * k * np.tanh(k * h) * (1 + gamma * k**2 / (rho * g)))

    # def w(k):
    # return np.sqrt( g * k * np.tanh(k * h) )

    # Cosa de energía
    a = st_fft_shifted  #* st_fft_filter
    a = a[M // 2:, :]
    a = a[:, N // 2:]

    # X, Y = np.meshgrid(np.linspace(-k_ny, k_ny, N), np.linspace(-w_ny, w_ny, M))
    # plt.pcolormesh(X, Y, np.log(np.abs(a)), shading='auto')

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel('$\omega$ [$s^{-1}$]', fontsize=22)
    ax.set_ylabel('$k$ [$m^{-1}$]', fontsize=22)

    k = np.linspace(0, k_ny, 1000)
    # plt.plot(1/0.01, 2*np.pi*17, 'xk')
    for i, hz in enumerate([10, 20, 30]):
        ax.axvline(2 * np.pi * hz, ls='--', c=BLACK_NORD)
        ax.text(2 * np.pi * hz + 10 - 5,
                50 + 110 * i,
                f'$\omega \simeq {2*np.pi*hz:.0f}$ $({hz}$hz)',
                c=BLACK_NORD,
                fontsize=16,
                bbox={
                    'boxstyle': 'square',
                    'color': WHITE_NORD
                })

    ax.plot(w_gam(k), k, c=RED_NORD, lw=2, label='Relación de disperción a orden lineal')

    ax.imshow(np.log(np.abs(a)).T,
              extent=[
                  0,
                  w_ny,
                  0,
                  k_ny,
              ],
              cmap=cmap,
              origin='lower',
              aspect='auto')

    ax.set_ylim(0, 1e3)
    ax.set_xlim(0, 400)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.legend(fontsize=18, framealpha=0.5, fancybox=False, loc=1)

    plt.savefig(
        f'../../Informes/Labo7_presentacion/figs/dispersion_relation.pdf',
        bbox_inches='tight',
        transparent=True)

    plt.show()


def main():
    # med_folder = 'MED33 - Subida en voltaje - 0902'
    # med_folder = 'MED35 - Subida en voltaje - 0902'
    # med_folder = 'MED42 - Mod de fase, oscis estacionarios - 0909'
    med_folder = 'MED30 - Subida en voltaje, NOTA - 0902'
    # med_folder = 'MED64 - Bajada en voltaje - 1104'
    st_diagram = get_st_diagram(med_folder, error_filter=5)
    # st_diagram = get_st_envelope(st_diagram)
    # st_diagram[st_diagram < 2] = 0
    # st_diagram = np.roll(st_diagram, 400, axis=1)

    # plt.imshow(st_diagram, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()

    dispersion_relation(st_diagram)
    # plt.figure()
    # plt.plot(st_diagram[1067], '-r')
    # plt.plot(st_diagram[1081,1700:1850], '-g')
    # plt.plot(st_diagram[1095], '-b')
    # st_diagram = get_st_spatial_envelope(st_diagram)
    # plt.plot(st_diagram[1067], '-r')
    # plt.plot(st_diagram[1081], '-g')
    # plt.plot(st_diagram[1095], '-b')
    # plt.colorbar()
    # plt.show()

    # get_envelope(st_diagram[2801], plot=True)
    # plot_st_envelope(st_diagram)
    # st_left, st_right = get_st_left_right(st_diagram, plot=True)

    # -- Plot 3D --
    # fig = go.Figure(data=[go.Surface(z=st_left[::10, ::10])])
    # fig.update_layout(title='ST')
    # fig.show()

    # -- Plot 2D --
    # st_lines = -st_diagram[:100].T
    # plt.plot(st_lines)
    # plt.show()
    # vmax = np.nanmax(st_diagram)
    # plt.imshow(st_diagram, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    # plt.colorbar()
    # plt.show()

    # -- Funciones --
    # med_start, med_end = 30, 35
    # meds_folder = '../../Mediciones/'
    # meds_to_process = [
    # p for p in sorted(os.listdir(meds_folder)) if 'MED' in p and
    # med_start <= int(re.findall(r'MED(\d+) - ', p)[0]) <= med_end
    # ]

    # amps = np.zeros(len(meds_to_process))
    # for i, med_folder in enumerate(meds_to_process):
    # st_diagram = get_st_diagram(med_folder, error_filter=False)
    # envelope = get_st_spatial_envelope(st_diagram)[1000:2000]
    # print(i)
    # amps[i] = envelope.max() - envelope.min()

    # plt.plot(amps**2, 'o')
    # plt.show()
    # vmax = np.max(st_diagram)
    # plt.imshow(envelope, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
