import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d
import scipy.fft as fft

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

    signal = uniform_filter1d(signal, size=signal.size//1000)
    peaks, _ = find_peaks(signal, distance=max_0dist*0.8, height=0)

    xp, yp = peaks, signal[peaks]

    yp_ext = np.concatenate((yp, yp[:1]))
    xp_ext = np.concatenate((xp, xp[:1]+signal.size))
    envelope = CubicSpline(xp_ext, yp_ext, bc_type='periodic')

    x = dom_signal
    y = envelope(x)

    # y = np.concatenate((y, y))
    # x = np.arange(y.size)

    # x = np.arange(ext_signal.size)
    # y = envelope(x)

    # fig, ax = plt.subplots(1, figsize=(12, 8), sharex=True, sharey=True)

    # ax.plot(xp_ext, yp_ext, 'or')
    # ax.plot(xp, yp, 'ok')
    # ax.plot(x, y, '-g')

    # ax.plot(dom_signal, signal, '-r', zorder=0)

    # ax.grid()
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
    st_spatial_envelope = get_st_spatial_envelope(st_diagram)
    st_envelope = get_st_envelope(st_diagram)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), sharex=True, sharey=True)

    img = ax1.imshow(st_diagram)
    fig.colorbar(img, ax=ax1)

    img = ax2.imshow(st_spatial_envelope)
    fig.colorbar(img, ax=ax2)

    img = ax3.imshow(st_envelope)
    fig.colorbar(img, ax=ax3)
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
    st_envelope = get_st_envelope(st_diagram)
    
    N, M = st_envelope.shape

    st_fft = fft.fft2(st_envelope)
    st_fft_shifted = fft.fftshift(st_fft)

    st_fft_left_filter, st_fft_right_filter = np.zeros_like(st_fft_shifted), np.zeros_like(st_fft_shifted)

    st_fft_left_filter[N//2:, M//2:] = 1
    st_fft_right_filter[N//2:, :M//2] = 1

    st_fft_left = fft.ifftshift(st_fft_left_filter*st_fft_shifted)
    st_fft_right = fft.ifftshift(st_fft_right_filter*st_fft_shifted)
    st_left, st_right = fft.ifft2(st_fft_left), fft.ifft2(st_fft_right)

    a = st_fft_shifted*(st_fft_left_filter + st_fft_right_filter)
    plt.imshow(np.log(np.abs(a)))
    plt.colorbar()
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)

    # img = ax1.imshow(np.real(st_left))
    # fig.colorbar(img, ax=ax1)

    # ax2.imshow(np.real(st_right))
    # fig.colorbar(img, ax=ax2)

    # plt.show()


def main():
    med_folder = '../../Mediciones/MED41 - Mod de fase - 0909/'
    hdf5_folder = med_folder+'HDF5/'
    f = h5py.File(hdf5_folder+'ST.hdf5', 'r')

    st_diagram = -np.array(f['spatiotemporal_diagram'])
    get_st_left_right(st_diagram)

    # plot_st_envelope(st_diagram)

    # st_error = np.array(f['spatiotemporal_diagram_error'])

    # line = st_diagram[2500, :]
    # get_envelope(line)

if __name__ == "__main__":
    main()
