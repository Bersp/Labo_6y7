from collections import namedtuple

from PyEMD import EMD
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from more_itertools import sliding_window
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate, find_peaks, welch
from scipy.stats import mode
from skimage.feature import canny
from mpl_utils import *

from envelope_analysis import (
    get_envelope,
    get_st_envelope,
    get_st_left_right,
    get_st_spatial_envelope,
)
from spatiotemporal_analysis import get_st_diagram

Med = namedtuple('Medicion', ['med_folder', 'good_spatial_line_idx', 'osc_limits'])

med40 = Med(med_folder='MED40 - Oscilones a full - 0902',
            good_spatial_line_idx=2057, osc_limits=None)

med69 = Med(med_folder='MED69 - Diversion - 1104', good_spatial_line_idx=2466, osc_limits=slice(1960, 2080))

med39 = Med(med_folder='MED39 - Subida en voltaje - 0902',
            good_spatial_line_idx=1041, osc_limits=None)

med36 = Med(med_folder='MED36 - Subida en voltaje - 0902',
            good_spatial_line_idx=1218, osc_limits=None)

med28 = Med(med_folder='MED28 - Subida en voltaje - 0902',
            good_spatial_line_idx=937, osc_limits=None)

med64 = Med(med_folder='MED64 - Bajada en voltaje - 1104',
            good_spatial_line_idx=570, osc_limits=None)


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


def fit_sech(x, A, lam, C, D):
    return A / np.cosh(lam * x + C) + D


def fit_osc(real_osc: np.ndarray, plot: bool = False):
    '''
    TODO: Docstring for fit_reflected_osc.

    Parameters
    ----------
    real_osc : TODO
    direction : TODO

    Returns
    -------
    TODO

    '''
    osc_argmax = np.argmax(real_osc)

    real_dom = np.arange(real_osc.size) - np.argmax(real_osc)
    popt, pcov = curve_fit(fit_sech, real_dom, real_osc)

    if plot:
        # scaled_real_dom = real_dom*2*np.pi/3000 # px -> rad

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(real_dom, real_osc, '.', color=DARK_GRAY_NORD, ms=5, label='Perfil de un oscilón')
        ax.plot(real_dom, fit_sech(real_dom, *popt), '-', color=BLUE_NORD, lw=3, label='Ajuste $f(x) = A\,sech(\lambda x) + D$')

        ax.set_title('Ajuste para un oscilón', fontsize=30, pad=35)
        ax.set_ylabel('Altura [u. a.]', fontsize=18)
        ax.set_xlabel('Ángulo [u. a.]', fontsize=18)
        # ax.set_xticks(np.linspace(0, 3000, 5, endpoint=True))
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.legend(fontsize=16, framealpha=0.5, loc=2)

        ax.grid()

        plt.savefig(
            f'../../Informes/Labo7_presentacion/figs/fit_one_osc.pdf',
            bbox_inches='tight',
            transparent=True)

        plt.show()

    return lambda x: fit_sech(x, *popt)


def fit_all_oscilons(spatial_line, ax=None):
    '''
    TODO: Docstring for fit_all_oscilons.

    Parameters
    ----------
    line : TODO

    Returns
    -------
    TODO

    '''
    if ax is None:
        ax = plt.gca()

    dom = np.arange(spatial_line.size)

    min_peaks, _ = find_peaks(-spatial_line, height=0, distance=20, width=20)
    slices_osc = [slice(n, m, 1) for n, m in zip(min_peaks, min_peaks[1:])]

    ax.plot(dom, spatial_line, '.', c=DARK_GRAY_NORD, ms=5)

    colors = [BLUE_NORD, LIGHTER_BLUE_NORD]

    for i, slice_osc in enumerate(slices_osc):
        real_osc = spatial_line[slice_osc]
        dom_osc = dom[slice_osc]

        try:
            f = fit_osc(real_osc)
            dom_osc_centered = dom_osc - np.argmax(real_osc) - slice_osc.start

            ax.plot(dom_osc, f(dom_osc_centered), c=colors[i%2], lw=3)
        except:
            pass

    ax.set_ylabel('Altura [u. a.]', fontsize=18)
    # ax.set_xlabel('Ángulo [rad]', fontsize=18)
    # ax.set_xticks(np.linspace(0, 3000, 5, endpoint=True))
    # ax.set_xticklabels(['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$', '$2\pi$'], fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=18)



def all_oscilons_overlaped(spatial_line, direction):
    '''
    TODO: Docstring for all_oscilons_overlaped.

    Parameters
    ----------
    line : TODO

    Returns
    -------
    TODO

    '''
    dom = np.arange(spatial_line.size)

    min_peaks, _ = find_peaks(-spatial_line, height=0, distance=20, width=20)
    slices_osc = [slice(n, m, 1) for n, m in zip(min_peaks, min_peaks[1:])]

    all_osc = np.array([])
    dom = np.array([])
    for i, slice_osc in enumerate(slices_osc):
        if i > 1:
            real_osc = spatial_line[slice_osc]
            real_osc /= real_osc.max()

            osc_argmax = np.argmax(real_osc)

            if direction == -1:
                half_osc = real_osc[:osc_argmax]
                reflected_osc = np.append(half_osc, half_osc[::-1][1:])

            elif direction == 1:
                half_osc = real_osc[osc_argmax:]
                reflected_osc = np.append(half_osc[::-1][1:], half_osc)

            all_osc = np.append(all_osc, reflected_osc)
            dom = np.append(
                dom,
                np.arange(reflected_osc.size) - np.argmax(reflected_osc))

    popt, pcov = curve_fit(fit_sech, dom, all_osc)

    plt.plot(dom, all_osc, '.')
    plt.plot(dom, fit_sech(dom, *popt), '.')

    plt.show()


def get_speed_direction(st_diagram,
                        good_spatial_line_idx,
                        plot_st=False,
                        plot_hist=False):
    '''
    TODO: Docstring for get_speed_direction.

    Parameters
    ----------
    st_diagram : TODO
    good_spatial_line_idx: TODO

    Returns
    -------
    TODO

    Notes
    -----
    MED69: good_spatial_line_idx=2466
    MED50: good_spatial_line_idx=2057
    '''
    M, N = st_diagram.shape
    st_diagram_10hz = st_diagram[good_spatial_line_idx %
                                 25::25]  # 25fps == 10hz

    # for i, (sl1, sl2) in enumerate(sliding_window(st_diagram_10hz[:20], 2)):
    # c = correlate(sl2, sl1, mode='same')
    # plt.plot(c, c=f'C{i}', alpha=0.3)
    # plt.plot(c.argmax(), c.max(), 's', c=f'C{i}', alpha=0.5)
    # plt.show()

    c = []
    for sl1, sl2 in sliding_window(st_diagram_10hz, 2):
        c.append(correlate(sl2, sl1, mode='same').argmax() - 1500)
    c = np.array(c)

    # idx = np.where(np.abs(c) < np.mean(c)+5*np.std(c))[0]
    c = c / 25
    c = c[np.abs(c) > 2]
    cmode = mode(c)[0][0]
    c = c[np.abs(c) > np.abs(cmode * .7)]
    c = c[np.abs(c) < np.abs(cmode * 1.3)]
    v = 1 / np.mean(c)

    if plot_st:
        cmap = truncate_colormap('twilight_shifted', 0.1, 0.90)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        extent = (0, N, 0, M)
        img = ax1.imshow(st_diagram, origin='lower', clim=(-7.5, 10), extent=extent, aspect=N/M, cmap=cmap)
        # fig.colorbar(img)

        axins = ax1.inset_axes([0.3, 0.5, 0.68, 0.55])
        axins.imshow(st_diagram,
                     extent=extent,
                     origin="lower",
                     clim=(-7.5, 10),
                     cmap=cmap,
                     aspect=N/M)

        x1, x2, y1, y2 = 1700, 2350, 2300, 2700
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        tag_ax(axins, 'Zoom', pos=(0.05, 0.9), box_color=cmap(0.5))

        for fl in np.arange((1348-750*v)%250, M, 250):
            def f(x): return v * x + fl
            x = np.arange(st_diagram.shape[1])
            ax1.plot(x, f(x), ':', c='0.2', lw=3)
            axins.plot(x, f(x), ':', c='0.2', lw=3)
        ax1.plot(x, f(x), ':', c='0.2', lw=3, label='Trayectoria de los trenes')

        ax1.set_xlabel('Ángulo [rad]', fontsize=18, labelpad=-10)
        ax1.set_ylabel('Tiempo [s]', fontsize=18, labelpad=-14)
        ax1.set_xticks([750, N])
        ax1.set_xticklabels(['$0$', '$2\pi$'], fontsize=16)
        ax1.set_yticks([750, M])
        ax1.set_yticklabels(['$0$', '$12$'], fontsize=16)

        ax1.set_xlim(750, N)
        ax1.set_ylim(750, M)


        ax1.legend(fontsize=16, fancybox=False, loc=4)

        ax2.hist(c*2*np.pi/N/(12/M), bins=9, ec=WHITE_NORD, color=DARK_GRAY_NORD)
        ax2.set_xlabel('Velocidad [rad/s]', fontsize=18)
        ax2.set_ylabel('Frecuencia', fontsize=18)

        plt.savefig(
            f'../../Informes/Labo7_presentacion/figs/velocidad_osc.pdf',
            bbox_inches='tight',
            transparent=True)

        plt.show()

    return v


def empirical_mode_decomposition(signal, plot=False):
    """
    TODO: Docstring for empirical_mode_decomposition.
    
    Parameters
    ----------
    signal :  TODO
    
    Returns
    -------
    TODO
    
    """

    t = np.linspace(0, 1, signal.size)
    IMF = EMD().emd(signal, t)
    N = IMF.shape[0] + 2

    if plot:
        # for n, imf in enumerate(IMF):
        # plt.subplot(N,1,n+2)
        # plt.plot(t, imf, 'g')
        # plt.title("IMF "+str(n+1))
        # plt.xlabel("Time [s]")

        # plt.subplot(N, 1, n+3)

        plt.plot(t, IMF[5] + IMF[4] + IMF[3], 'g')
        plt.plot(t, IMF[4] + IMF[3], 'b')
        plt.plot(t, signal, ':r')
        plt.title("Recompuesta")
        plt.xlabel("Time [s]")

        plt.show()

    return IMF[3:6].sum(0)


def st_empirical_mode_decomposition(st_diagram):
    '''
    TODO: Docstring for st_empirical_mode_decomposition.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    '''
    new_st_diagram = np.zeros_like(st_diagram)

    for i, line in enumerate(st_diagram):
        print(i)
        new_st_diagram[i] = empirical_mode_decomposition(line)

    return new_st_diagram


def main():
    # med_folder = 'MED69 - Diversion - 1104'
    # med_folder = 'MED40 - Oscilones a full - 0902'
    med = med69
    st_diagram = get_st_diagram(med.med_folder, error_filter=5)

    # plt.imshow(st_diagram, origin='lower')
    # plt.colorbar()

    # a = 1/5
    # x = np.linspace(0, 3000)
    # plt.plot(x, a*x+2306, c='r')
    # plt.xlim(0, 3000)
    # plt.show()

    # for i, l in enumerate(st_diagram[2466::25]):
    # if i >= 5:
    # break
    # plt.plot(l, label=i)
    # plt.legend()
    # plt.show()

    # spatial_line = st_diagram[2466]
    # st_left, st_right = get_st_left_right(st_diagram)

    # plt.imshow(st_right, origin='lower')
    # plt.colorbar()

    # ----------------------------------------------------------------------
    # EMD
    # empirical_mode_decomposition(spatial_line)
    # st_empirical_mode_decomposition(st_diagram)

    # ----------------------------------------------------------------------
    # Speed direction
    # v = get_speed_direction(st_diagram, med.good_spatial_line_idx, True, False)
    # csum = csum % 3000
    # plt.plot(csum, d, '.r')
    # plt.show()

    # ----------------------------------------------------------------------
    # Func for spatial lines
    spatial_line = st_diagram[med.good_spatial_line_idx]
    # fit_all_oscilons(spatial_line)
    fit_osc(spatial_line[med.osc_limits], plot=True) # med69
    # plt.plot(spatial_line, c='r')
    plt.show()
    # all_oscilons_overlaped(spatial_line, -1)


if __name__ == '__main__':
    main()
