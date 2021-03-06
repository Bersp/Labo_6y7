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
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['figure.figsize'] = [16, 8]


color_array = [NORD10, NORD12, NORD14, NORD11, NORD15]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_array)
