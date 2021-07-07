import numpy as np
import skimage.measure as skm
import skimage.io as sio
import skimage.filters as sif
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

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

    labeled = skm.label(imth, connectivity=2)
    objects = skm.regionprops(labeled)

    props = [(object.label, object.area, object.bbox, object.perimeter) for object in objects]
    #  def sort_key_function(p): # TODO: Sortear adecuadamente el cuadradito
        #  min_row, min_col, max_row, max_col = p[2]
        #  return abs((max_row-min_row)/(max_col-min_col) - 1.2/1.9)
        #  return abs(p[3]/p[1] - 2*(1.2+1.9)/(1.2 * 1.9))
    #  print('\n'.join([f'{p[0]}\t{sort_key_function(p)}' for p in props]))

    label_annulus = sorted(props, key=lambda p: p[1], reverse=True)[1][0] # NOTE: Cambiar ese 1 por un 0
    annulus = labeled == label_annulus
    filtered_annulus = sif.roberts(annulus)
    threshold = get_threshold(filtered_annulus)
    filtered_annulus = filtered_annulus > threshold

    labeled = skm.label(filtered_annulus, connectivity=2)
    objects = skm.regionprops(labeled)
    props = [(object.label, object.area) for object in objects]
    (label_circ1, _), (label_circ2, _) = sorted(props, key=lambda p: p[1], reverse=False)[:2]

    XY = np.vstack(np.where(labeled==label_circ1)).T
    x0, y0, r = taubin_svd(XY)
    plt.plot([x0, y0], [x0+r, y0], 'r.')

    XY = np.vstack(np.where(labeled==label_circ2)).T
    x0, y0, r = taubin_svd(XY)
    plt.plot([x0, y0], [x0+r, y0], 'b.')

    plt.imshow(labeled, cmap='gray')
    #  plt.plot([x0, y0], [x0+r, y0], 'ro')

    #  plt.show()
    #  plt.imshow(labeled==label_annulus, cmap='gray')
    #  plt.figure()
    #  plt.imshow(labeled==565, cmap='gray')
    plt.show()

    return

    # sort regionprops by area*solidity*extent, in ascending order



    mask_of_disk_alone = ( labeled == props[-1][1] )
    mask_of_rect_alone = ( labeled == props[-2][1] )
    mask_of_annulus_alone = ( labeled == props[-3][1] )
    if modificado==True:
        ### OJO QUE ACA LO MODIFIQUE A MANO para 09/10
        mask_of_rect_alone = ( labeled == props[-3][1] )
        mask_of_annulus_alone = ( labeled == props[-5][1] )

    # Relleno la mascara del anillo por si quedaron agujeros
    mask_of_annulus_alone = dilation(mask_of_annulus_alone, disk(2))
    xc_disk, yc_disk, R_disk = determine_properties_of_disk(mask_of_disk_alone)
    xc_ext_annulus, yc_ext_annulus, Rext_annulus = determine_properties_of_disk(mask_of_annulus_alone)
    xc_int_annulus, yc_int_annulus, Rint_annulus  = determine_properties_of_disk(mask_int_annulus(mask_of_annulus_alone))

    # Both the internal and external fit of the annulus have the same center
    xc_annulus, yc_annulus = np.mean([xc_ext_annulus, xc_int_annulus]), np.mean([yc_ext_annulus, yc_int_annulus])

    R_disk = R_disk + 4 # this only affects the R_disk variable, not the mask itself
    Rint_annulus = Rint_annulus + 5
    Rext_annulus = Rext_annulus + 5

    xc_rect, yc_rect, sl_rect = determine_properties_of_rect(mask_of_rect_alone)

    mask = ( mask_of_disk_alone + mask_of_rect_alone + mask_of_annulus_alone)

    return mask, [xc_disk, yc_disk], R_disk, [xc_rect, yc_rect], sl_rect, [xc_annulus, yc_annulus], Rext_annulus, Rint_annulus, mask_of_disk_alone, mask_of_rect_alone, mask_of_annulus_alone

if __name__ == '__main__':
    img = get_white_img('../../Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/gray/Camera procimage500_eagle-117361-00001.tiff')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED2 - 0624/gray/ID_0_C1S0001000001.tif')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED2 - 0624/white/IM_C1S0001000001.tif')
    #  img = get_white_img('../../Mediciones_FaradayWaves/MED0 - Medicion con 16 bits - 0611/gray/IM_C1S0003000001.tif')

    generate_mask(img)
