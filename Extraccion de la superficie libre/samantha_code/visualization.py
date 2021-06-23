import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import reset_output

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

def show_html_image(field_2d, output_file_name='test.html'):
    """
    Plot a 2D field as image using Bokeh, output to html file only.
    """

    # Needs resetting, otherwise it appends new result to old one
    reset_output()

    # Define ranges
    xR, yR = field_2d.shape
    x = np.linspace(0, xR-1, xR)
    y = np.linspace(0, yR-1, yR)
    # Generate figure canvas with tooltips
    p = figure(x_range=(0, 1), y_range=(0, 1), \
           tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], \
           match_aspect=True)
    # Generate image
    p.image(image=[field_2d], x=0, y=0, dw=1, dh=1, \
            palette="Spectral11")

    # Output to html file
    output_file(output_file_name, title="visualization")

    # Display in browser
    show(p)
    return None


def create_csv_datafile(field, filename):
    """
    Creates a csv file in "X Y Z" format, suitable for Paraview plotting.
    """
    Lx, Ly = field.shape
    xyz = np.zeros((Lx*Ly,3))
    k = 0
    for i in range(Lx):
        for j in range(Ly):
            xyz[k,:] = [i, j, field[i, j]]
            k = k + 1
    np.savetxt(filename, xyz, delimiter=",")
    return None
    
