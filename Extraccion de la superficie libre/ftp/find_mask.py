import numpy as np
import skimage.measure as skm
import skimage.io as sio
import skimage.filters as sif
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy import ndimage

def taubin_svd(XY):
    """
    algebraic circle fit
    input: list [[x_1, y_1], [x_2, y_2], ....]
    output: a, b, r.  a and b are the center of the fitting circle, and r is the radius
     Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """
    def old_div(a, b): return a/b
    X = XY[:,0] - np.mean(XY[:,0]) # norming points by x avg
    Y = XY[:,1] - np.mean(XY[:,1]) # norming points by y avg
    centroid = [np.mean(XY[:,0]), np.mean(XY[:,1])]
    Z = X * X + Y * Y
    Zmean = np.mean(Z)
    Z0 = old_div((Z - Zmean), (2. * np.sqrt(Zmean)))
    ZXY = np.array([Z0, X, Y]).T
    U, S, V = np.linalg.svd(ZXY, full_matrices=False) #
    V = V.transpose()
    A = V[:,2]
    A[0] = old_div(A[0], (2. * np.sqrt(Zmean)))
    A = np.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
    a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
    r = np.sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/abs(A[0])/2;
    return a,b,r

def get_white_img(filename):
    img = sio.imread(filename).astype(float)
    return img

def get_threshold(image):

    y, bins = np.histogram(image.flatten(), bins='doane')
    x = (bins[1:] + bins[:-1])/2

    y_diff = np.gradient(y)
    idx_max = argrelextrema(y, np.greater)[0] # determina los máximos locales

    # Toma los dos índices de los máximos más grandes
    idx_max = [y[i] if i in idx_max else 0 for i in range(len(y))]
    idx_max2, idx_max1 = np.argsort(idx_max)[-2:]

    y_diff = y_diff[idx_max1: idx_max2]

    idx_min = np.argsort(np.abs(y_diff))[0]
    return x[idx_min]

def generate_mask(white_img):

    threshold = get_threshold(white_img)
    imth = white_img > threshold

    # Busco el annulus
    labeled = skm.label(imth, connectivity=2)
    objects = skm.regionprops(labeled)

    props = [(object.label, object.area) for object in objects]

    label_annulus = sorted(props, key=lambda p: p[1], reverse=True)[0][0]
    annulus_mask = labeled == label_annulus

    # Busco el cuadradito adentro del radio externo del annulus
    filled_annulus = ndimage.binary_fill_holes(annulus_mask)
    inner_annulus_area = imth*filled_annulus*(1-annulus_mask)

    labeled = skm.label(inner_annulus_area, connectivity=1)
    objects = skm.regionprops(labeled)
    props = [(object.label, object.area, object.bbox, object.perimeter)
             for object in objects]

    def sort_key_function(p): # TODO: Sortear adecuadamente el cuadradito
        min_row, min_col, max_row, max_col = p[2]
        return abs((max_row-min_row)/(max_col-min_col) - 1.2/1.9)

    label_square = sorted(props, key=sort_key_function)[0][0]

    square_mask = labeled == label_square

    return annulus_mask, square_mask

def get_annulus_radii(annulus_mask):

    # Consigo propiedades
    filtered_annulus = sif.roberts(annulus_mask)
    threshold = get_threshold(filtered_annulus)
    filtered_annulus = filtered_annulus > threshold

    labeled = skm.label(filtered_annulus, connectivity=2)
    objects = skm.regionprops(labeled)
    props = [(object.label, object.area) for object in objects]
    (label_circ1, _), (label_circ2, _) = sorted(props, key=lambda p: p[1], reverse=False)[:2]

    XY = np.vstack(np.where(labeled==label_circ1)).T
    *_, r_inner = taubin_svd(XY)

    XY = np.vstack(np.where(labeled==label_circ2)).T
    x0, y0, r_outer = taubin_svd(XY)

    #  plt.plot([x0, y0], [x0+r_inner, y0], 'r.')
    #  plt.plot([x0, y0], [x0+r_outer, y0], 'b.')

    #  plt.imshow(labeled, cmap='gray')
    #  plt.show()

    return x0, y0, r_inner, r_outer

if __name__ == '__main__':
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/gray/Camera procimage500_eagle-117361-00001.tiff')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED2 - 0624/gray/ID_0_C1S0001000001.tif')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED2 - 0624/white/IM_C1S0001000001.tif')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED0 - Medicion con 16 bits - 0611/gray/IM_C1S0003000001.tif')
    img = get_white_img('../../Mediciones_FaradayWaves/MED5 - No medimos - 0716/white/ID_0_C1S0001000001.tif')

    masks = generate_mask(img)
    circle_props = get_annulus_radii(masks[0])
    np.save('../samantha_code/FTP_toy/MED5_masks', masks)
    np.save('../samantha_code/FTP_toy/MED5_circle_props', circle_props)
