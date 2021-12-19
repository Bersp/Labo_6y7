from collections import namedtuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from envelope_analysis import get_st_envelope
import mpl_utils
from spatiotemporal_analysis import get_st_diagram
from oscilons_analysis import fit_all_oscilons


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


Med = namedtuple('Medicion', ['name', 'med_folder', 'clim', 'subreg'])

med40 = Med('Oscilones fuertes',
            'MED40 - Oscilones a full - 0902',
            clim=(-6, 6),
            subreg=(1800, 2300, 950, 1450))

med69 = Med('Oscilones fuertes',
            'MED69 - Diversion - 1104',
            clim=(-7.5, 7.5),
            subreg=(1700, 2200, 1200, 1700))

med36 = Med('Oscilones suaves',
            'MED36 - Subida en voltaje - 0902',
            clim=None,
            subreg=(1550, 2050, 1300, 1800))

med39 = Med('Oscilones suaves',
            'MED39 - Subida en voltaje - 0902',
            clim=None,
            subreg=None)

med64 = Med('Nose', 'MED64 - Bajada en voltaje - 1104', clim=None, subreg=None)

med41 = Med('Modulación de fase',
            'MED41 - Mod de fase - 0909',
            clim=None,
            subreg=(1200, 1700, 400, 900))


def taylor_couette():
    med_folder = 'MED30 - Subida en voltaje, NOTA - 0902'
    st_diagram = get_st_diagram(med_folder, error_filter=5)
    st_diagram = get_st_envelope(st_diagram)

    extent = [0, 2 * np.pi, 0, 3072]

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(st_diagram,
                    origin='lower',
                    extent=extent,
                    cmap=cmap,
                    aspect=2 * np.pi / 3072)

    fig.colorbar(img, fraction=0.0454)

    ax.set_xlabel('Ángulo [rad]', fontsize=18, labelpad=-10)
    ax.set_ylabel('Tiempo [s]', fontsize=18, labelpad=-14)
    ax.set_xticks([0, 2 * np.pi])
    ax.set_xticklabels(['$0$', '$2\pi$'], fontsize=16)
    ax.set_yticks([0, 3072])
    ax.set_yticklabels(['$0$', '$12$'], fontsize=16)

    plt.savefig(
        f'../../Informes/Labo7_presentacion/figs/st_taylor_couette.pdf',
        bbox_inches='tight',
        transparent=True)

def plot_3ax_fit_all_osc():
    meds_folder = ['MED69 - Diversion - 1104', 'MED64 - Bajada en voltaje - 1104', ]
    gsp_idxs = [2466, 570]

    fig, axes = plt.subplots(2, figsize=(12, 9))
    for med_folder, gsp_idx, ax in zip(meds_folder, gsp_idxs, axes):
        st_diagram = get_st_diagram(med_folder, error_filter=5)
        spatial_line = st_diagram[gsp_idx]

        fit_all_oscilons(spatial_line, ax=ax)
        ax.set_xticks(np.linspace(0, 3000, 5, endpoint=True))
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.grid()

    axes[0].set_title('Ajuste para todos los oscilones', fontsize=30, pad=35)
    ax.set_xlabel('Ángulo [rad]', fontsize=18)
    ax.set_xticks(np.linspace(0, 3000, 5, endpoint=True))
    ax.set_xticklabels(['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$', '$2\pi$'], fontsize=16)


    plt.savefig(
        f'../../Informes/Labo7_presentacion/figs/fit_all_osc.pdf',
        bbox_inches='tight',
        transparent=True)


def plot_one_med(med):

    st_diagram = get_st_diagram(med.med_folder, error_filter=5)

    extent = [0, 2 * np.pi, 0, 3072]

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(st_diagram,
                    clim=med.clim,
                    origin='lower',
                    extent=extent,
                    cmap=cmap,
                    aspect=2 * np.pi / 3072)
    fig.colorbar(img, fraction=0.0454)

    # inset axes....
    if med.subreg is not None:
        subreg = list(med.subreg)
        subreg[:2] = map(lambda n: n * 2 * np.pi / 3000, subreg[:2])
        axins = ax.inset_axes([0.5, 0.515, 0.47, 0.47])
        axins.imshow(st_diagram,
                     extent=extent,
                     origin="lower",
                     clim=med.clim,
                     cmap=cmap,
                     aspect=2 * np.pi / 3072)

        # ax.set_title(med.name, fontsize=30, pad=35)
        ax.set_xlabel('Ángulo [rad]', fontsize=18, labelpad=-10)
        ax.set_ylabel('Tiempo [s]', fontsize=18, labelpad=-14)
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xticklabels(['$0$', '$2\pi$'], fontsize=16)
        ax.set_yticks([0, 3072])
        ax.set_yticklabels(['$0$', '$12$'], fontsize=16)
        # ax.set_yticklabels([4 ,5], 'uf')

        # sub region of the original image
        x1, x2, y1, y2 = subreg
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])

    # if med.clim is None:
        # min_st, max_st = st_diagram.min(), st_diagram.max()
    # else:
        # min_st, max_st = med.clim

    # max_patch = mpatches.Patch(color=cmap(1 - 1e-15),
                               # label=f'{max_st:.2f} u. a.')
    # min_patch = mpatches.Patch(color=cmap(0), label=f'{min_st:.2f} u. a.')
    # ax.legend(handles=[min_patch, max_patch],
              # fontsize=18,
              # loc=2,
              # fancybox=False)

    med_name = med.name.replace(' ', '_').lower()
    plt.savefig(
        f'../../Informes/Labo7_presentacion/figs/st_med_example.pdf',
        bbox_inches='tight',
        transparent=True)


def main():
    plot_one_med(med36)
    # plot_one_med(med69)
    # plot_one_med(med41)

    # taylor_couette()

    # plot_3ax_fit_all_osc()

    plt.show()


if __name__ == '__main__':
    main()
