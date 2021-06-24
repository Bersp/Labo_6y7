import numpy as np
from modulation_instability.ftp import calculate_phase_diff_map_1D
from modulation_instability.fringe_extrapolation import gerchberg2d
from modulation_instability.utils import polar2cart, img2polar, sum_values, find_first_and_last, find_index_of_false_values, divisor
import numpy.ma as ma

def individual_ftp(kk, args):
    dset_ftp_d, resfactor, gray, mask_g, ref_m_gray, N_iter_max, n, th, c_rect, sl_rect, mask_of_annulus_alone, lin_min_idx_a,lin_max_idx_a, col_min_idx_a, col_max_idx_a = args
    # dset_ftp_d, resfactor, gray, mask_g, ref_m_gray, N_iter_max, n, th, c_rect, sl_rect, kk = args
    def_image = dset_ftp_d[:, :, kk]
    # 1. Undistort image
    # def_image = undistort_image(def_image, mapx, mapy)
    # 2. Substract gray
    def_m_gray = def_image - resfactor*gray
    #ELIMINO LOS REFLEJOS A MANO
    #def_m_gray[(def_m_gray<-30)] = 0
    #def_m_gray[(def_m_gray>30)] = 0

    # 3. Extrapolate fringes
    def_m_gray = gerchberg2d(def_m_gray, mask_g, N_iter_max=N_iter_max)

    # 4. Process by FTP
    dphase = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, \
        th, n)

    dphase_rect = np.mean( dphase[(c_rect[0]-sl_rect):(c_rect[0]+sl_rect), \
            (c_rect[1]-sl_rect):(c_rect[1]+sl_rect)] )

    # 5a). Time unwrapping using rectangle area

    # 5b). Substract solid movement
    dphase = dphase - dphase_rect

    # 6. Calculate height field
    height = dphase#height_map_from_phase_map(dphase, L, D, pspp)
    height = (ma.getdata(height)*mask_of_annulus_alone)[lin_min_idx_a:lin_max_idx_a, col_min_idx_a:col_max_idx_a]
    return height

def height_annulus_polar_coordinates(frame, mask_annulus, phase_width = 3000):
    """
    Returns a matrix with the height of the annulus in polar coordinates.
    """
    # mask_annulus = mask[:,:,0]
    annulus = frame
    annulus_polar = img2polar(ma.masked_array(annulus, mask=(1-mask_annulus)))
    first_idx, last_idx = find_index_of_false_values(annulus_polar)
    if first_idx != last_idx:
        annulus_polar_lines = ma.getdata(annulus_polar[first_idx:last_idx,:]) #por si toma mal la mascara
        prom = np.average(annulus_polar_lines, axis=0)
    else:
        annulus_polar_lines = ma.getdata(annulus_polar[first_idx,:])
        prom = annulus_polar_lines
    return prom

def polar_coordinates(kk, args):
    deformed, mask_annulus = args
    frame = deformed[:,:,kk]
    prom = height_annulus_polar_coordinates(frame, mask_annulus)
    return prom
