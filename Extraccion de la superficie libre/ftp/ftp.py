from itertools import repeat
import json
import logging
from multiprocessing import Pool, cpu_count

import h5py
from numba import njit
import numpy as np
import numpy.ma as ma
from scipy import ndimage
from scipy import signal
from scipy.optimize import curve_fit
import skimage.filters as sif
import skimage.measure as skm
from skimage.morphology import dilation, disk
from unwrap import unwrap

# Logger config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')


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


def individual_ftp(deformed, gray, filled_ref, annulus_mask, annulus_center,
                   annulus_radii):
    """
    TODO: Docstring for individual_ftp
    """

    deformed = deformed - gray

    # Frankestein
    r_inner, r_outer = annulus_radii
    x0, y0 = annulus_center
    width = r_outer - r_inner
    r_middle = (r_outer + r_inner) // 2
    mask_out = tukey_2d(x0, y0, 1024, r_middle, width * 2, int(width))
    mask_in = tukey_2d(x0, y0, 1024, r_middle, width, int(width))
    frankestein = deformed * (mask_in) + filled_ref * (1 - mask_out)

    dphase = calculate_phase_diff_map_1d(frankestein,
                                         filled_ref,
                                         th=0.9,
                                         ns=3,
                                         mask_for_unwrapping=1 - annulus_mask)

    #  for i in range(8): annulus_mask = binary_erosion(annulus_mask)
    dphase[annulus_mask == 0] = np.nan

    return dphase


def _gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def gerchberg2d(interferogram, mask_where_fringes_are, N_iter_max):
    """
    TODO: Docstring for gerchberg2d
    """

    ref = interferogram
    refh = interferogram * mask_where_fringes_are
    interf = mask_where_fringes_are

    ft_ref = np.fft.rfft2(ref)
    ft_refh = np.fft.rfft2(refh)

    S = ref.shape
    S = S[0]

    y = (np.abs(ft_refh[0, :]))
    y = y / np.max(y)
    x = np.linspace(0, (len(y) - 1), len(y))
    maxInd = signal.argrelextrema(y, np.greater)
    x, y = x[maxInd], y[maxInd]
    n = len(x)
    w = signal.hann(n)
    y = y * w
    index_mean = np.argwhere(y == np.max(y))[0, 0]
    mean = maxInd[0][index_mean]
    sigma = np.sum(y * (x - mean)**2) / n
    try:
        popt, pcov = curve_fit(_gauss,
                               x,
                               y,
                               p0=[y[index_mean], mean, sigma],
                               maxfev=1100)
    except:
        popt, pcov = curve_fit(_gauss, x, y, maxfev=1100)

    k0x, k0y = popt[1], 0
    R_in_k_space = popt[2]  # *2.5

    kx, ky = np.meshgrid(range(int(S / 2 + 1)), range(S))

    cuarto_superior = ((kx - k0x)**2 + (ky - (S - k0y))**2 <= R_in_k_space**2)
    cuarto_inferior = ((kx - k0x)**2 + (ky - (0 - k0y))**2 <= R_in_k_space**2)
    lugar_a_conservar = cuarto_inferior + cuarto_superior
    lugar_a_anular = 1 - lugar_a_conservar

    lugar_a_anular = lugar_a_anular.nonzero()
    interf = interf.nonzero()

    En = np.zeros(N_iter_max + 1)

    ii = 0
    while ii <= N_iter_max:
        ft_refh[lugar_a_anular] = 0
        refhc = np.fft.irfft2(ft_refh)
        refhc[interf] = refh[interf]
        ft_refh = np.fft.rfft2(refhc)
        En[ii] = np.sum(np.abs(ft_refh))
        if ii > 0 and En[ii - 1] < En[ii]:
            break
        ii += 1
    En = En[0:ii]

    refhc = np.real(refhc)
    refhc[interf] = ref[interf]

    return refhc


def tukey_2d(x0, y0, L, R, A, D):
    """
    TODO: Docstring for tukey_2d
    construimos una ventana de tukey en 2D
    L = imagen resultante en tamaño L x L
    R = radio donde inicia la ventana de Tukey
    A = longitud del plateau
    D = longitud de crecimiento/decrecimiento
    """
    output = np.zeros((L, L))
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    x -= x0 - L // 2
    y -= y0 - L // 2
    r = np.sqrt((x - L // 2)**2 + (y - L // 2)**2)
    region_plateau = (r >= (R - A // 2)) * (r <= (R + A // 2))
    region_subida = (r >= (R - A // 2 - D)) * (r < (R - A // 2))
    region_bajada = (r >= (R + A // 2)) * (r < (R + A // 2 + D))
    output[region_plateau] = 1
    output[region_subida] = 0.5 * (1 -
                                   np.cos(np.pi / D *
                                          (r[region_subida] - R - A // 2 - D)))
    output[region_bajada] = 0.5 * (1 + np.cos(np.pi / D *
                                              (r[region_bajada] - R + A // 2)))

    return output


class FTP():
    """
    TODO: Docstring for FTP
    """
    def __init__(self, hdf5_folder):

        self.hdf5_folder = hdf5_folder

        f = h5py.File(hdf5_folder + 'RAW.hdf5')

        self.deformed = f['deformed']
        self.gray = np.asarray(f['gray'])
        self.reference = np.asarray(f['reference'])
        self.white = np.asarray(f['white'])

        camera_attrs = json.loads(f.attrs['CAMERA'])
        self.img_resolution = camera_attrs['resolution']
        self.n_deformed_images = camera_attrs['n_deformed_images']

        calibration_attrs = json.loads(f.attrs['CALIBRATION'])
        self._L, self._d = calibration_attrs['L'], calibration_attrs['d']

        self._annulus_mask = None
        self._square_mask = None

        self._annulus_center = None
        self._annulus_radii = None

        self._filled_ref = None

        self.output_chunks_shape = self.deformed.chunks

    # -------------------------------------------------------------------------
    # Properties

    @property
    def square_mask(self):
        if self._square_mask is not None:
            return self._square_mask
        else:
            return self._generate_masks()[1]

    @property
    def annulus_mask(self):
        if self._annulus_mask is not None:
            return self._annulus_mask
        else:
            return self._generate_masks()[0]

    @property
    def annulus_center(self):
        if self._annulus_center is not None:
            return self._annulus_center
        else:
            return self._get_annulus_props()[0]

    @property
    def annulus_radii(self):
        if self._annulus_radii is not None:
            return self._annulus_radii
        else:
            return self._get_annulus_props()[1]

    # -------------------------------------------------------------------------
    # FTP functions

    @property
    def filled_ref(self):
        if self._filled_ref is not None:
            return self._filled_ref
        else:
            mask_where_fringes_are = self.annulus_mask + self.square_mask
            reference = self.reference - self.gray
            self._filled_ref = gerchberg2d(reference, mask_where_fringes_are,
                                           100)
            return self._filled_ref

    def chunk_ftp(self, deformed_chunk):
        dphase_chunk = np.zeros(deformed_chunk.shape)
        n_images = deformed_chunk.shape[2]

        ftp_args = (self.gray, self.filled_ref, self.annulus_mask,
                    self.annulus_center, self.annulus_radii)

        for i in range(n_images):

            dphase = individual_ftp(deformed_chunk[:, :, i], *ftp_args)
            dphase_chunk[:, :, i] = dphase

            if i == 0:
                logging.info(
                    f'FTP: {1}/{n_images} imágenes del chunk calculadas')
            elif i > 1 and i % 20 == 0:
                logging.info(
                    f'FTP: {i}/{n_images} imágenes del chunk calculadas')

        return dphase_chunk

    # -------------------------------------------------------------------------
    # Mask and annulus props functions

    def _generate_masks(self):

        if self._annulus_mask is not None and self._square_mask is not None:
            return self._annulus_mask, self._square_mask

        imth = self._get_image_thresholded(self.white)

        # Busco el annulus
        labeled = skm.label(imth, connectivity=2)
        objects = skm.regionprops(labeled)

        props = [(object.label, object.area) for object in objects]

        label_annulus = sorted(props, key=lambda p: p[1], reverse=True)[0][0]
        annulus_mask = labeled == label_annulus

        # Busco el cuadradito adentro del radio externo del annulus
        filled_annulus = ndimage.binary_fill_holes(annulus_mask)
        inner_annulus_area = imth * filled_annulus * (1 - annulus_mask)

        labeled = skm.label(inner_annulus_area, connectivity=1)

        objects = skm.regionprops(labeled)
        props = [(object.label, object.area, object.bbox)
                 for object in objects]

        def sort_key_function(p):
            min_row, min_col, max_row, max_col = p[2]
            return abs((max_row - min_row) / (max_col - min_col) - 1.2 / 1.9)

        aspect_relation_filter = sorted(props, key=sort_key_function)[:10]
        label_square = sorted(aspect_relation_filter,
                              key=lambda p: p[1],
                              reverse=True)[0][0]

        self._annulus_mask = annulus_mask
        self._square_mask = labeled == label_square
        return self._annulus_mask, self._square_mask

    def _get_image_thresholded(self, image):

        y, bins = np.histogram(image.flatten(), bins='doane')
        x = (bins[1:] + bins[:-1]) / 2

        y_diff = np.gradient(y)
        # determina los máximos locales
        idx_max = signal.argrelextrema(y, np.greater)[0]

        # Toma los dos índices de los máximos más grandes
        idx_max = [y[i] if i in idx_max else 0 for i in range(len(y))]
        idx_max2, idx_max1 = np.argsort(idx_max)[-2:]

        y_diff = y_diff[idx_max1:idx_max2]

        idx_min = np.argsort(np.abs(y_diff))[0]
        image_threshold = image > x[idx_min]

        return image_threshold

    def _get_annulus_props(self):

        filtered_annulus = sif.roberts(self.annulus_mask)
        filtered_annulus = self._get_image_thresholded(filtered_annulus)

        labeled = skm.label(filtered_annulus, connectivity=2)
        objects = skm.regionprops(labeled)
        props = [(object.label, object.area) for object in objects]
        (label_circ1, _), (label_circ2, _) = sorted(props,
                                                    key=lambda p: p[1],
                                                    reverse=False)[:2]

        XY = np.vstack(np.where(labeled == label_circ1)).T
        *_, r_inner = self._taubin_svd(XY)

        XY = np.vstack(np.where(labeled == label_circ2)).T
        v0, h0, r_outer = self._taubin_svd(XY)

        self._annulus_center = (int(v0), int(h0))
        self._annulus_radii = (int(r_inner), int(r_outer))
        return self._annulus_center, self._annulus_radii

    def _taubin_svd(self, XY):
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
        def old_div(a, b):
            return a / b

        X = XY[:, 0] - np.mean(XY[:, 0])  # norming points by x avg
        Y = XY[:, 1] - np.mean(XY[:, 1])  # norming points by y avg
        centroid = [np.mean(XY[:, 0]), np.mean(XY[:, 1])]
        Z = X * X + Y * Y
        Zmean = np.mean(Z)
        Z0 = old_div((Z - Zmean), (2. * np.sqrt(Zmean)))
        ZXY = np.array([Z0, X, Y]).T
        U, S, V = np.linalg.svd(ZXY, full_matrices=False)
        V = V.transpose()
        A = V[:, 2]
        A[0] = old_div(A[0], (2. * np.sqrt(Zmean)))
        A = np.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
        a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
        r = np.sqrt(A[1] * A[1] + A[2] * A[2] - 4 * A[0] * A[3]) / abs(
            A[0]) / 2
        return a, b, r

    # -------------------------------------------------------------------------
    # Reference fringe data
    def get_fringes_physical_size(self, mm_d_inner=205, mm_d_outer=215):
        """
        Parameters
        ----------
        mm_d_inner : Diámetro interno del anillo en mm
        mm_d_outer : Diámetro externo del anillo en mm

        Returns
        -------
        p : Longitud de onda del patrón proyectado en mm
        """

        ref_masked = self.reference * self.square_mask

        ref_masked[ref_masked == 0] = np.nan
        ref_line = np.nanmean(ref_masked, 0)
        ref_line = ref_line[~np.isnan(ref_line)]

        peaks, _ = signal.find_peaks(ref_line)
        peaks_distance = np.diff(peaks[1:-2]).mean()  # sin los de las puntas

        px_r_inner, px_r_outer = self.annulus_radii
        mm_px_rate = np.mean(
            [mm_d_inner / (2 * px_r_inner), mm_d_outer / (2 * px_r_outer)])

        p = peaks_distance * mm_px_rate
        return p

    # -------------------------------------------------------------------------
    # Export functions

    def export(self):
        f = h5py.File(self.hdf5_folder + 'FTP.hdf5', 'w')

        f.attrs.create('p', self.get_fringes_physical_size())
        f.attrs.create('L', self._L)
        f.attrs.create('d', self._d)

        height_grp = f.create_group('height_fields')
        masks_grp = f.create_group('masks')

        self._get_mask_and_export(masks_grp)
        f.flush()
        self._do_ftp_and_export_height_fields(height_grp)

        f.close()
        logging.info(f'FTP: END\n')

    def _get_mask_and_export(self, masks_grp):
        masks_grp_annulus = masks_grp.create_dataset('annulus',
                                                     shape=self.img_resolution,
                                                     dtype='float64')
        masks_grp_annulus[:, :] = self.annulus_mask
        masks_grp_annulus.attrs['center'] = self.annulus_center
        masks_grp_annulus.attrs['annulus_radii'] = self.annulus_radii

        masks_grp_square = masks_grp.create_dataset('square',
                                                    shape=self.img_resolution,
                                                    dtype='float64')
        masks_grp_square[:, :] = self.square_mask

    def _do_ftp_and_export_height_fields(self, height_grp):

        height_grp.create_dataset('annulus',
                                  shape=(*self.img_resolution,
                                         self.n_deformed_images),
                                  chunks=self.output_chunks_shape,
                                  dtype='float64')

        img_per_chunk = self.output_chunks_shape[2]
        n_chunks = np.ceil(self.n_deformed_images / img_per_chunk).astype(int)

        logging.info(f'FTP: Inicio del proceso')

        for i in range(n_chunks):
            chunk = (img_per_chunk * i, img_per_chunk * (i + 1))

            deformed_chunk = self.deformed[:, :, chunk[0]:chunk[1]]

            height_field_chunk = self.chunk_ftp(deformed_chunk)

            height_grp['annulus'][:, :, chunk[0]:chunk[1]] = height_field_chunk
            logging.info(f'FTP: {i+1}/{n_chunks} chunks guardados')
        logging.info(f'END')


# if __name__ == '__main__':
import matplotlib.pyplot as plt

med_folder = '../../Mediciones/MED69 - Diversion - 1104/'
hdf5_folder = med_folder + 'HDF5/'

ftp = FTP(hdf5_folder)

# ftp_args = (ftp.gray, ftp.filled_ref, ftp.annulus_mask,
            # ftp.annulus_center, ftp.annulus_radii)
# dphase = individual_ftp(ftp.deformed[:,:,10], *ftp_args)

plt.imshow(ftp.annulus_mask)
plt.colorbar()
plt.show()

    # print(ftp.get_fringes_physical_size())

    # ir, ar = ftp.annulus_radii
    # v0, h0 = ftp.annulus_center

    # plt.imshow(mask)
    # plt.plot(h0 + ir, v0, 'r.')
    # plt.plot(h0 + ar, v0, 'r.')
    # plt.show()

    # ftp.export()

    # f = h5py.File(hdf5_folder+'FTP.hdf5', 'r')
    # img = f['height_fields']['annulus'][:, :, 0]
    # img -= np.nanmean(img)
    # plt.imshow(img, clim=(-10, -30))
    # plt.colorbar()
    # plt.show()
