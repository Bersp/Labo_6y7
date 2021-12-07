import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.fft as fft
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns

from spatiotemporal_analysis import get_st_diagram
# sns.set_palette(sns.color_palette("rocket", 5))


def get_zeros(signal):
    a = np.isclose(signal, 0, atol=5*np.mean(np.abs(np.diff(signal))))
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


def get_envelope(signal):
    dom_signal = np.arange(signal.size)

    max_0dist = np.max(np.diff(
        np.where(np.isclose(signal, 0, atol=1.1*np.abs(np.min(np.diff(signal)))))))

    if max_0dist == 1:
        max_0dist += 1

    signal = uniform_filter1d(signal, size=signal.size//1000)
    peaks, _ = find_peaks(signal, distance=max_0dist*0.8, height=0)

    xp, yp = peaks, signal[peaks]

    yp_ext = np.concatenate((yp, yp[:1]))
    xp_ext = np.concatenate((xp, xp[:1]+signal.size))
    envelope = CubicSpline(xp_ext, yp_ext, bc_type='periodic')

    x = dom_signal
    y = envelope(x)

    # fig, ax = plt.subplots(1, figsize=(14, 8), sharex=True, sharey=True)

    # ax.plot(xp, yp, 'ok', label='Máximos')
    # ax.plot(x, y, '-', c='indianred', label='Envolvente')

    # ax.plot(dom_signal, signal, '--k', zorder=0, label='Señal original')

    # ax.set_xlabel('Longitud de arco [u. a.]', fontsize=16)
    # ax.set_ylabel('Amplitud [mm]', fontsize=16)

    # ax.grid()
    # ax.legend(fontsize=14, loc='upper center', ncol=3, framealpha=0.2)

    # plt.savefig('/home/bersp/env_1sample.pdf', dpi=300,
                # bbox_inches='tight', transparent=True)
    # plt.show()

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
        img = ax.imshow(diag, cmap='coolwarm')
        ax.set_title(title, fontsize=16)
        # fig.colorbar(img, ax=ax1)

    st_spatial_envelope = get_st_spatial_envelope(st_diagram)
    st_envelope = get_st_envelope(st_diagram)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 8), sharex=True, sharey=True
    )

    plot_aux(ax1, st_diagram, 'Diagrama original')

    plot_aux(ax2, st_spatial_envelope, 'Envolvente espacial')

    plot_aux(ax3, st_envelope, 'Envolvente espacio-temporal')

    ax1.set_xticks([])
    # ax1.set_yticks([])
    # plt.savefig('/home/bersp/st_envelopes.pdf', dpi=300,
                # bbox_inches='tight', transparent=True)

    plt.show()


def get_st_left_right(st_diagram):
    """
    TODO: Docstring for get_st_left_right.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    """
    st_envelope = get_st_spatial_envelope(st_diagram)
    # st_envelope = st_diagram

    N, M = st_envelope.shape

    st_fft = fft.fft2(st_envelope)
    st_fft_shifted = fft.fftshift(st_fft)

    st_fft_left_filter = np.zeros_like(st_fft_shifted)
    st_fft_right_filter = np.zeros_like(st_fft_shifted)
    st_fft_left_filter[N//2:, M//2:] = 1
    st_fft_right_filter[N//2:, :M//2] = 1

    st_fft_left = fft.ifftshift(st_fft_left_filter*st_fft_shifted)
    st_fft_right = fft.ifftshift(st_fft_right_filter*st_fft_shifted)
    st_left, st_right = fft.ifft2(st_fft_left), fft.ifft2(st_fft_right)

    # Cosa de energía
    # a = st_fft_shifted*(st_fft_left_filter + st_fft_right_filter)
    # plt.imshow(np.log(np.abs(a)))
    # plt.colorbar()
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(
    # 1, 2, figsize=(12, 8), sharex=True, sharey=True
    # )

    # vmax = np.max(np.real(st_left))*.5
    # img = ax1.imshow(np.real(st_left), cmap='coolwarm')#, vmin=-vmax, vmax=vmax)
    # ax1.set_title('Izquierda', fontsize=16)
    # fig.colorbar(img, ax=ax1)

    # vmax = np.max(np.real(st_right))*.5
    # ax2.imshow(np.real(st_right), cmap='coolwarm')#, vmin=-vmax, vmax=vmax)
    # ax2.set_title('Derecha', fontsize=16)
    # fig.colorbar(img, ax=ax2)

    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # plt.savefig('/home/bersp/st_left_right.pdf', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

    return np.real(st_left), np.real(st_right)

def main():
    med_folder = 'MED69 - Diversion - 1104'
    st_diagram = get_st_diagram(med_folder, error_filter=5)
    plt.imshow(st_diagram)
    # plt.figure()
    # plt.plot(st_diagram[1067], '-r')
    # plt.plot(st_diagram[1081,1700:1850], '-g')
    # plt.plot(st_diagram[1095], '-b')
    # st_diagram = get_st_spatial_envelope(st_diagram)
    # plt.plot(st_diagram[1067], '-r')
    # plt.plot(st_diagram[1081], '-g')
    # plt.plot(st_diagram[1095], '-b')
    # plt.colorbar()
    plt.show()
    
    # get_envelope(st_diagram[300])
    # plot_st_envelope(st_diagram)
    # st_left, st_right = get_st_left_right(st_diagram)

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

    # get_st_left_right(st_diagram)


if __name__ == "__main__":
    main()
