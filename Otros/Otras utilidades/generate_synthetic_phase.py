import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from unwrap import unwrap

from mpl_utils import tag_ax


def calculate_phase_diff_map_1d(dY, dY0, th, ns, mask_for_unwrapping=None):
    """
    TODO: Docstring for calculate_phase_diff_map_1d
    # Basic FTP treatment.
    # This function takes a deformed and a reference image and calculates the phase difference map between the two.
    #
    # INPUTS:
    # dY	= deformed image
    # dY0	= reference image
    # ns	= size of gaussian filter
    #
    # OUTPUT:
    # dphase 	= phase difference map between images
    """

    ny, nx = np.shape(dY)
    phase0 = np.zeros([nx, ny])
    phase = np.zeros([nx, ny])

    for lin in range(0, nx):
        fY0 = np.fft.fft(dY0[lin, :])
        fY = np.fft.fft(dY[lin, :])

        dfy = 1. / ny
        fy = np.arange(dfy, 1, dfy)

        imax = np.argmax(np.abs(fY0[9:nx // 2]))
        ifmax = imax + 9

        HW = np.round(ifmax * th)
        HW *= 0.5  # TODO
        W = 2 * HW
        win = signal.tukey(int(W), ns)

        gaussfilt1D = np.zeros(nx)
        gaussfilt1D[int(ifmax - HW - 1):int(ifmax - HW + W - 1)] = win

        Nfy0 = fY0 * gaussfilt1D
        Nfy = fY * gaussfilt1D

        Ny0 = np.fft.ifft(Nfy0)
        Ny = np.fft.ifft(Nfy)

        phase0[lin, :] = np.angle(Ny0)
        phase[lin, :] = np.angle(Ny)

    if mask_for_unwrapping is None:
        mphase0 = unwrap(phase0)
        mphase = unwrap(phase)
    else:
        mphase0 = ma.masked_array(phase0, mask=mask_for_unwrapping)
        mphase = ma.masked_array(phase, mask=mask_for_unwrapping)
        mphase0 = unwrap(mphase0)
        mphase = unwrap(mphase)

    dphase = (mphase - mphase0)
    return dphase


def _gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def main():
    v = np.linspace(-1, 1, 1024)
    x, y = np.meshgrid(v, v)
    phase_imposed = (6 * (1 - x)**2. * np.exp(-(x**2) - (y + 1)**2) - 20 *
                     (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2) -
                     1 / 3 * np.exp(-(x + 1)**3 - y**4)) / 2
    phase_imposed *= 1

    f0 = 80
    im_ref = np.sin(f0 * x)
    im_def = np.sin(f0 * x + phase_imposed)

    good_dphase = calculate_phase_diff_map_1d(im_def, im_ref, th=0.9, ns=3)

    fig, axes = plt.subplots(3, figsize=(5, 10))
    ax1, ax2, ax3 = axes

    ax1.imshow(im_ref, cmap='gray')
    ax2.imshow(im_def, cmap='gray')
    ax3.imshow(good_dphase, cmap='coolwarm')

    for ax in axes:
        ax.set_xticks([])
    for ax in axes:
        ax.set_yticks([])

    for ax, tag in zip(axes, ['Referencia', 'Deformada', 'Altura']):
        tag_ax(ax, tag, pos=(0.06, 0.90), fontsize=14)

    for ax in axes: ax.patch.set_alpha(0.)

    plt.savefig('../../Informes/Labo7_presentacion/figs/synthetic_ftp.png',
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
