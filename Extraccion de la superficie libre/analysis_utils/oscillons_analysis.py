import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, welch, correlate
from skimage.feature import canny
from PyEMD import EMD
from more_itertools import sliding_window


from envelope_analysis import get_envelope, get_st_envelope, get_st_spatial_envelope, get_st_left_right
from spatiotemporal_analysis import get_st_diagram


def fit_sech(x, A, lam, C, D):
    return A / np.cosh(lam * x + C) + D


def fit_reflected_osc(real_osc: np.ndarray,
                      direction: int,
                      plot: bool = False):
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

    # if direction == -1:
        # half_osc = real_osc[:osc_argmax]
        # osc_tail = real_osc[osc_argmax:]
        # reflected_osc = np.append(half_osc, half_osc[::-1][1:])
        # reflected_dom = np.arange(-half_osc.size, half_osc.size - 1)
        # tail_dom = np.arange(0, osc_tail.size)

    # elif direction == 1:
        # half_osc = real_osc[osc_argmax:]
        # osc_tail = real_osc[:osc_argmax]
        # reflected_osc = np.append(half_osc[::-1][:-1], half_osc)
        # reflected_dom = np.arange(-half_osc.size, half_osc.size - 1)
        # tail_dom = np.arange(-osc_tail.size, 0)

    real_dom = np.arange(real_osc.size) - np.argmax(real_osc)
    popt, pcov = curve_fit(fit_sech, real_dom, real_osc)

    if plot:
        plt.plot(reflected_dom, reflected_osc, 'ok')
        plt.plot(reflected_dom, fit_sech(reflected_dom, *popt), '-r')
        plt.plot(tail_dom, osc_tail, 'x', c='gray')

        plt.show()

    return lambda x: fit_sech(x, *popt)


def fit_all_oscilons(spatial_line, direction):
    '''
    TODO: Docstring for fit_all_oscilons.

    Parameters
    ----------
    line : TODO
    direction : TODO

    Returns
    -------
    TODO

    '''
    dom = np.arange(spatial_line.size)

    min_peaks, _ = find_peaks(-spatial_line, height=0, distance=20, width=20)
    slices_osc = [slice(n, m, 1) for n, m in zip(min_peaks, min_peaks[1:])]

    plt.plot(dom, spatial_line, '.k')
    for i, slice_osc in enumerate(slices_osc):
        real_osc = spatial_line[slice_osc]
        dom_osc = dom[slice_osc]

        f = fit_reflected_osc(real_osc, direction)
        dom_osc_centered = dom_osc - np.argmax(real_osc) - slice_osc.start

        plt.plot(dom_osc, f(dom_osc_centered), c=f'C{i%2}', lw=3)

    # plt.show()


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


def get_speed_direction(st_diagram):
    '''
    TODO: Docstring for get_speed_direction.

    Parameters
    ----------
    st_diagram : TODO

    Returns
    -------
    TODO

    '''

    c = []
    for sl1, sl2 in sliding_window(st_diagram, 2):
        c.append(correlate(sl2, sl1, mode='same').argmax() - 1500)
    c = np.array(c)
    d = np.arange(c.size)

    idx = np.where(np.abs(c) < np.mean(c)+2*np.std(c))[0]

    csum = np.cumsum(c[idx])
    d = d[idx]

    return d, csum
    # plt.plot(np.cumsum(c))
    # plt.plot(d, c, 'r')
    # plt.plot(d[idx], c[idx], 'b')
    # plt.show()

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
    N = IMF.shape[0]+2

    if plot:
        # for n, imf in enumerate(IMF):
            # plt.subplot(N,1,n+2)
            # plt.plot(t, imf, 'g')
            # plt.title("IMF "+str(n+1))
            # plt.xlabel("Time [s]")

        # plt.subplot(N, 1, n+3)

        plt.plot(t, IMF[5]+IMF[4]+IMF[3], 'g')
        plt.plot(t, IMF[4]+IMF[3], 'b')
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
    med_folder = 'MED69 - Diversion - 1104'
    st_diagram = get_st_diagram(med_folder, error_filter=5)
    emd_st_diagram = np.load('new_st_diagram.npy')
    # spatial_line = st_diagram[2466]
    st_left, st_right = get_st_left_right(st_diagram)

    plt.imshow(st_right, origin='lower')
    plt.colorbar()
    


    # ----------------------------------------------------------------------
    # EMD
    # empirical_mode_decomposition(spatial_line)
    # st_empirical_mode_decomposition(st_diagram)

    # ----------------------------------------------------------------------
    # Speed deirection
    d, c = get_speed_direction(st_right)
    c = c % 3000
    plt.plot(c, d, 'r')
    plt.show()

    # ----------------------------------------------------------------------
    # Func for spatial lines
    # fit_all_oscilons(spatial_line, 1)
    # spatial_line = st_diagram[2476]
    # plt.plot(spatial_line, c='r')
    # plt.show()
    # all_oscilons_overlaped(spatial_line, -1)


if __name__ == '__main__':
    main()
