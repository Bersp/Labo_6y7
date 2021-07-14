import h5py
import numpy as np
import skimage.measure as skm
import warnings

from scipy.signal import argrelextrema

class FTP():

    def __init__(self, hdf5_folder):

        self.hdf5_folder = hdf5_folder

        #  f = h5py.File(hdf5_folder+'RAW.hdf5')

        #  raw_images = f['raw_images']
        #  self.gray = raw_images['gray']
        #  self.white = raw_images['white']
        #  self.deformed = raw_images['deformed']
        #  self.reference = raw_images['reference']

        #  self.img_resolution = raw_images.attrs['img_resolution']
        #  self.n_deformed_images = raw_images.attrs['n_deformed_images']

        self._annulus_area = None
        self._square_area = None

        self._annulus_mask = None
        self._square_mask = None
        self._annulus_center = None

        self.output_chunks_shape = (64, 64, 100)

    # TODO: Limpiar el area de la máscara del annulus
    @property
    def annulus_mask(self):
        if self._annulus_mask is not None:
            return self._annulus_mask
        else:
            return self._get_annulus_mask()[0]

    @property
    def annulus_center(self):
        if self._annulus_center is not None:
            return self._annulus_center
        else:
            return self._get_annulus_mask()[1]

    # TODO: Limpiar el area de la máscara del square
    @property
    def square_mask(self):
        if self._square_mask is not None:
            return self._square_mask
        else:
            return self._get_square_mask()[1]

    # TODO: Bueno... Implementar.
    def ftp(self, deformed_chunk):
        warnings.warn('Función ftp no implementada, devuelve las deformadas\n', RuntimeWarning)
        return deformed_chunk


    def export(self):
        f = h5py.File(self.hdf5_folder+'FTP.hdf5', 'w')

        height_grp = f.create_group('height_fields')
        masks_grp = f.create_group('masks')

        height_grp.create_dataset('annulus',
                  shape=(*self.img_resolution, self.n_deformed_images),
                  chunks=self.output_chunks_shape, dtype='float64')

        self._get_mask_and_export(masks_grp)
        self._do_ftp_and_export_height_fields(height_grp)

        f.flush()
        f.close()

    # TODO: Implementar
    def _get_annulus_mask(self):
        """Trata el área detectada para el annulus y retorna una máscara y su centro"""
        warnings.warn('Función _get_annulus_mask no implementada, retorna directamente el area detectada y el centro harcodeado\n', RuntimeWarning)
        return self._find_areas_of_interest()[0], [self.img_resolution[0]//2]*2

    # TODO: Implementar
    def _get_square_mask(self):
        warnings.warn('Función _get_square_mask no implementada, retorna directamente el area detectada\n', RuntimeWarning)
        """Trata el área detectada para el square y retorna su máscara"""
        return self._find_areas_of_interest()[1]

    def _get_mask_and_export(self, masks_grp):
        masks_grp_annulus = masks_grp.create_dataset('annulus', shape=self.img_resolution,
                                  dtype='float64')
        masks_grp_annulus[:, :] = self.annulus_mask
        masks_grp_annulus.attrs['center'] = self.annulus_center

        masks_grp_square = masks_grp.create_dataset('square', shape=self.img_resolution,
                                  dtype='float64')
        masks_grp_square[:, :] = self.square_mask

    def _do_ftp_and_export_height_fields(self, height_grp):
        img_per_chunk = self.output_chunks_shape[2]
        n_chunks = self.n_deformed_images//img_per_chunk

        for i in range(n_chunks):
            deformed_chunk = self.deformed[:, :, img_per_chunk*i:img_per_chunk*(i+1)]
            height_field_chunk = self.ftp(deformed_chunk)
            height_grp['annulus'][:, :, img_per_chunk*i:img_per_chunk*(i+1)] = height_field_chunk

    def _find_areas_of_interest(self):

        if self._annulus_area is not None and self._square_area is not None:
            return self._annulus_area, self._square_area

        white = np.asarray(self.white).mean(2)
        white_threshold = self._get_image_thresholded(white)

        labeled = skm.label(white_threshold, connectivity=2)
        objects = skm.regionprops(labeled)

        # NOTE: Test en el labo con Pablo
        #  props = [(object.label, object.area, object.bbox, object.perimeter) for object in objects]
        #  def sort_key_function(p): # TODO: Sortear adecuadamente el cuadradito
            #  min_row, min_col, max_row, max_col = p[2]
            #  return abs((max_row-min_row)/(max_col-min_col) - 1.2/1.9)
            #  return abs(p[3]/p[1] - 2*(1.2+1.9)/(1.2 * 1.9))
        #  print('\n'.join([f'{p[0]}\t{sort_key_function(p)}' for p in props]))

        props = [(object.label, object.area) for object in objects]
        label_annulus = sorted(props, key=lambda p: p[1], reverse=True)[1][0]
        self._annulus_area = labeled == label_annulus
         # NOTE: Cambiar ese 1 por un 0 si no se trabaja con imagenes con el circulo central lleno

        self._square_area = np.zeros(shape=self.img_resolution) # TODO: Falta calcular y devolver el square_area
        return self._annulus_area, self._square_area            # TODO: ver dilation(_, disk) para rellenar pixels muertos en el disco

    def _get_image_thresholded(self, image):
        return image > ( np.mean(image) + 0.2*np.std(image) )

        y, bins = np.histogram(image.flatten(), bins='doane')
        x = (bins[1:] + bins[:-1])/2

        y_diff = np.gradient(y)
        idx_max = argrelextrema(y, np.greater)[0] # determina los máximos locales

        # Toma los dos índices de los máximos más grandes
        idx_max = [y[i] if i in idx_max else 0 for i in range(len(y))]
        idx_max2, idx_max1 = np.argsort(idx_max)[-2:]

        y_diff = y_diff[idx_max1: idx_max2]

        idx_min = np.argsort(np.abs(y_diff))[0]
        image_threshold = image > x[idx_min]

        return image_threshold

    def _taubin_svd(XY): # NOTE: Por ahora esta func no está siendo utilizada
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

if __name__ == '__main__':
    ftp = FTP('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED42 - 0707/HDF5/')

    f = h5py.File('/media/box/Laboratorio/Labo_6y7/Mediciones_FaradayWaves/MED42 - 0707/HDF5/RAW.hdf5', 'r')

    raw_images = f['ftp_images']
    ftp.gray = raw_images['gray']
    ftp.white = raw_images['gray']
    ftp.deformed = raw_images['deformed']
    ftp.reference = raw_images['reference']

    ftp.img_resolution = (696, 696)
    ftp.n_deformed_images = 100

    ftp.export()
    #  import matplotlib.pyplot as plt
    #  fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    #  ax1.imshow(ftp.gray[:, :, 4], cmap='gray')
    #  mask = ftp.annulus_mask
    #  ax2.imshow(mask, alpha=0.5)
    #  plt.show()
