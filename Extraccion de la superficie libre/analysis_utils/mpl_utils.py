import matplotlib as mpl

NORD0 = '#2E3440'
NORD1 = '#3B4252'
NORD2 = '#434C5E'
NORD3 = '#4C566A'
NORD4 = '#D8DEE9'
NORD5 = '#E5E9F0'
NORD6 = '#ECEFF4'
NORD7 = '#8FBCBB'
NORD8 = '#88C0D0'
NORD9 = '#81A1C1'
NORD10 = '#5E81AC'
NORD11 = '#BF616A'
NORD12 = '#D08770'
NORD13 = '#EBCB8B'
NORD14 = '#A3BE8C'
NORD15 = '#B48EAD'

BLACK_NORD = NORD0
WHITE_NORD = NORD6
DARK_GRAY_NORD = NORD3
LIGHT_GRAY_NORD = NORD4

BLUE_GREEN_NORD = NORD7
LIGHTER_BLUE_NORD = NORD8
LIGHT_BLUE_NORD = NORD9
BLUE_NORD = NORD10
RED_NORD = NORD11
ORANGE_NORD = NORD12
YELLOW_NORD = NORD13
GREEN_NORD = NORD14
VIOLET_NORD = NORD15

# Matplotlib parameters
mpl.rcParams['grid.alpha'] = '0.4'
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['figure.figsize'] = [16, 8]

color_array = [NORD10, NORD12, NORD14, NORD11, NORD15]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_array)

# Useful functions


def tag_ax(ax,
           text,
           pos='topright',
           fontsize=16,
           text_color=BLACK_NORD,
           box_color=WHITE_NORD):

    if box_color == None:
        bbox_params = None

    else:
        bbox_params = {'boxstyle': 'square', 'color': box_color}

    if isinstance(pos, tuple):
        ax.text(*pos,
                text,
                fontweight='bold',
                fontsize=fontsize,
                va='center',
                ha='left',
                transform=ax.transAxes,
                c=text_color,
                bbox=bbox_params)
        return None

    if 'right' in pos:
        ha = 'right'
        x = 0.99
    elif 'left' in pos:
        ha = 'left'
        x = 0.01

    if 'top' in pos:
        va = 'top'
        y = 0.95
    elif 'bottom' in pos:
        va = 'bottom'
        y = 0.05

    ax.text(x,
            y,
            text,
            fontweight='bold',
            fontsize=fontsize,
            ha=ha,
            va=va,
            transform=ax.transAxes,
            c=text_color,
            bbox=bbox_params)
