import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from unwrap import unwrap

import h5py


def calculate_phase_diff_map_1D(dY, dY0, th, ns, mask_for_unwrapping=None):
    """
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

    import numpy.ma as ma

    ny, nx = np.shape(dY)
    phase0 = np.zeros([nx,ny])
    phase  = np.zeros([nx,ny])

    for lin in range(0, nx):
        fY0=np.fft.fft(dY0[lin,:])
        fY=np.fft.fft(dY[lin,:])

        dfy=1./ny
        fy=np.arange(dfy,1,dfy)

        imax=np.argmax(np.abs(fY0[9:nx//2]))
        ifmax=imax+9

        HW=np.round(ifmax*th)
        W=2*HW
        win=signal.tukey(int(W),ns)


        gaussfilt1D= np.zeros(nx)
        gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)]=win

        Nfy0 = fY0*gaussfilt1D
        Nfy = fY*gaussfilt1D

        Ny0=np.fft.ifft(Nfy0)
        Ny=np.fft.ifft(Nfy)

        phase0[lin,:] = np.angle(Ny0)
        phase[lin,:]  = np.angle(Ny)

    if mask_for_unwrapping is None:
        mphase0 = unwrap(phase0)
        mphase = unwrap(phase)
    else:
        mphase0 = ma.masked_array(phase0, mask=mask_for_unwrapping)
        mphase  = ma.masked_array(phase,  mask=mask_for_unwrapping)
        mphase0 = unwrap(mphase0)
        mphase = unwrap(mphase)

    dphase = (mphase-mphase0);
    return dphase

def tukey_2d(x0, y0, L, R, A, D):
    """
    construimos una ventana de tukey en 2D
    L = imagen resultante en tamaño L x L
    R = radio donde inicia la ventana de Tukey
    A = longitud del plateau
    D = longitud de crecimiento/decrecimiento
    """
    output = np.zeros((L,L))
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    x -= x0 - L//2
    y -= y0 - L//2
    r = np.sqrt( (x-L//2)**2 + (y-L//2)**2)
    region_plateau = (r>=(R-A//2)) * (r<=(R+A//2))
    region_subida = (r>=(R-A//2-D)) * (r<(R-A//2))
    region_bajada = (r>=(R+A//2)) * (r<(R+A//2+D))
    output[region_plateau] = 1
    output[region_subida] = 0.5*(1-np.cos(np.pi/D*(r[region_subida]-np.mean(r[r==(R-A//2-D)]))))
    output[region_bajada] = 0.5*(1+np.cos(np.pi/D*(r[region_bajada]-np.mean(r[r==(R+A//2)]))))

    return output

def test_syntetic_phase():
    v = np.linspace(-1, 1, 1024)
    x, y = np.meshgrid(v, v)
    phase_imposed = (3*(1-x)**2.*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2))/2

    f0 = 80
    im_ref = np.sin(f0*x)
    im_def = np.sin(f0*x + phase_imposed)

    mask_out = tukey_2d(1024, 300, 60, 30)
    mask_in = tukey_2d(1024, 300, 30, 30)

    good_dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(mask_in)
    #  good_dphase = phase_imposed*(mask_in)
    frankestein = im_def*(mask_in) + im_ref*(1-mask_out)
    im_def_by_pieces = im_def*(mask_in) + im_def*(1-mask_out)

    im_ref = im_ref*(mask_in) + im_ref*(1-mask_out)
    dphase_mask = calculate_phase_diff_map_1D(frankestein, im_ref, th=0.9, ns=3)

    good_dphase[mask_in != 1] = np.nan
    dphase_mask[mask_in != 1] = np.nan

    good_dphase -= np.nanmean(good_dphase)
    dphase_mask -= np.nanmean(dphase_mask)

    print(np.nanmax(good_dphase)-np.nanmin(good_dphase))
    print(np.nanmax(dphase_mask)-np.nanmin(dphase_mask))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey= True)
    axes = axes.flatten()

    mapp = axes[0].imshow(good_dphase)
    axes[0].set_title('good dphase')
    fig.colorbar(mapp, ax=axes[0])

    mapp = axes[1].imshow(dphase_mask)
    axes[1].set_title('dphase mask')
    fig.colorbar(mapp, ax=axes[1])

    mapp = axes[2].imshow(good_dphase-dphase_mask)
    axes[2].set_title('dphase diff')
    fig.colorbar(mapp, ax=axes[2])

    mapp = axes[3].imshow(frankestein)
    axes[3].set_title('frankestein')
    fig.colorbar(mapp, ax=axes[3])

    mapp = axes[4].imshow(im_def_by_pieces)
    axes[4].set_title('im_def_by_pieces')
    fig.colorbar(mapp, ax=axes[4])

    axes[5].get_shared_y_axes().remove(axes[5])
    axes[5].set_ylim(-0.5, 1.5)
    axes[5].plot(mask_in[mask_in.shape[0]//2, :], label='mask_in')
    axes[5].plot(mask_out[mask_in.shape[0]//2, :], label='mask_out')
    axes[5].legend()

    fig.subplots_adjust(right=0.8)

    plt.show()

def test_square_img():
    deformed = np.load('../FTP_toy/deformed.npy')
    reference  = np.load('../FTP_toy/reference.npy')

    inicial, final = 300,600
    im_ref = reference[inicial:final, inicial:final]
    im_def = deformed[inicial:final, inicial:final]

    mask_out = tukey_2d(im_ref.shape[0], 90, 30, 30)
    mask_in = tukey_2d(im_ref.shape[0], 90, 15, 30)

    #  good_dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(mask_in)
    frankestein = im_def*(mask_in) + im_ref*(1-mask_out)
    im_def_by_pieces = im_def*(mask_in) + im_def*(1-mask_out)

    im_ref = im_ref*(mask_in) + im_ref*(1-mask_out)
    dphase_mask = calculate_phase_diff_map_1D(frankestein, im_ref, th=0.9, ns=3)*(mask_in)

    good_dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(mask_in)

    good_dphase[mask_in != 1] = np.nan
    dphase_mask[mask_in != 1] = np.nan

    good_dphase -= np.nanmean(good_dphase)
    dphase_mask -= np.nanmean(dphase_mask)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey= True)
    axes = axes.flatten()

    mapp = axes[0].imshow(good_dphase)
    axes[0].set_title('good dphase')
    fig.colorbar(mapp, ax=axes[0])

    mapp = axes[1].imshow(dphase_mask)
    axes[1].set_title('dphase mask')
    fig.colorbar(mapp, ax=axes[1])

    mapp =axes[2].imshow(good_dphase-dphase_mask)
    axes[2].set_title('dphase diff')
    fig.colorbar(mapp, ax=axes[2])

    mapp =axes[3].imshow(frankestein)
    axes[3].set_title('frankestein')
    fig.colorbar(mapp, ax=axes[3])

    mapp =axes[4].imshow(im_def_by_pieces)
    axes[4].set_title('im_def_by_pieces')
    fig.colorbar(mapp, ax=axes[4])

    axes[5].get_shared_y_axes().remove(axes[5])
    axes[5].set_ylim(-0.5, 1.5)
    axes[5].plot(mask_in[mask_in.shape[0]//2, :], label='mask_in')
    axes[5].plot(mask_out[mask_in.shape[0]//2, :], label='mask_out')
    axes[5].legend()

    fig.subplots_adjust(right=0.8)

    plt.show()

def _gerchberg2d(interferogram, mask_where_fringes_are, N_iter_max):

    from scipy.signal import argrelextrema
    from scipy.optimize import curve_fit
    from scipy.signal import hann

    def gaus(x, a, x0, sigma): return a*np.exp(-(x-x0)**2/(2*sigma**2))

    ref = interferogram
    refh = interferogram*mask_where_fringes_are
    interf = mask_where_fringes_are

    ft_ref  = np.fft.rfft2(ref)
    ft_refh = np.fft.rfft2(refh)

    S = ref.shape
    S = S[0]

    y = (np.abs(ft_refh[0,:]))
    y = y/np.max(y)
    x = np.linspace(0, (len(y)-1), len(y))
    maxInd = argrelextrema(y, np.greater)
    x, y = x[maxInd], y[maxInd]
    n = len(x)
    w = hann(n)
    y = y*w
    index_mean = np.argwhere(y==np.max(y))[0,0]
    mean =  maxInd[0][index_mean]
    sigma = np.sum(y*(x-mean)**2)/n
    try:
        popt, pcov = curve_fit(gaus, x, y, p0 = [y[index_mean], mean, sigma],maxfev=1100)
    except:
        popt, pcov = curve_fit(gaus, x, y,maxfev=1100)

    k0x, k0y = popt[1], 0
    R_in_k_space = popt[2]#*2.5

    kx, ky = np.meshgrid(range(int(S/2+1)), range(S))

    cuarto_superior = ( (kx-k0x)**2 + (ky-(S-k0y))**2 <= R_in_k_space**2 )
    cuarto_inferior = ( (kx-k0x)**2 + (ky-(0-k0y))**2 <= R_in_k_space**2 )
    lugar_a_conservar = cuarto_inferior + cuarto_superior
    lugar_a_anular = 1-lugar_a_conservar

    lugar_a_anular = lugar_a_anular.nonzero()
    interf = interf.nonzero()

    En = np.zeros(N_iter_max+1)

    ii = 0
    while ii<=N_iter_max:
        ft_refh[lugar_a_anular] = 0
        refhc = np.fft.irfft2(ft_refh)
        refhc[interf] = refh[interf]
        ft_refh = np.fft.rfft2(refhc)
        En[ii] = np.sum(np.abs(ft_refh))
        if ii > 0 and En[ii-1] < En[ii]:
            break
        ii += 1
    En = En[0:ii]

    refhc = np.real(refhc)
    refhc[interf] = ref[interf]

    return refhc

def test_annulus_img():
    from skimage.morphology import dilation, disk

    deformed = np.load('../FTP_toy/deformed.npy')
    reference  = np.load('../FTP_toy/reference.npy')

    f = h5py.File('/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-annulus-PRO.hdf5', 'r')
    samantha_dphase = f['height_fields/annulus'][:, :, 17]

    f = h5py.File('/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED - Mediciones de Samantha para testear/HDF5/2018-07-17-0001-RAW.hdf5', 'r')


    gray = np.array(f['ftp_images']['gray']).mean(axis=2)
    deformed = f['ftp_images']['deformed'][:, :, 17]-gray
    reference = np.array(f['ftp_images']['reference']).mean(axis=2)-gray
    mask_where_fringes_are = np.zeros(reference.shape)
    mask_where_fringes_are[250:450, 150:550] = 1

    f = h5py.File('/home/bersp/Documents/Labo_6y7/Mediciones_FaradayWaves/MED42 - 0716/HDF5/FTP.hdf5', 'r')
    mask = np.array(f['masks']['annulus'])
    mask = dilation(mask, disk(2))

    mask_where_fringes_are += mask

    full_def = _gerchberg2d(deformed, mask_where_fringes_are, 100)
    full_ref = _gerchberg2d(reference, mask_where_fringes_are, 100)
    good_dphase = calculate_phase_diff_map_1D(full_def, full_ref, th=0.9, ns=3, mask_for_unwrapping=1-mask)

    good_dphase[mask == 0] = np.nan
    good_dphase = good_dphase[:-5, :-5]

    good_dphase -= np.nanmean(good_dphase)
    samantha_dphase -= np.nanmean(samantha_dphase)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    mapp = ax1.imshow(samantha_dphase, cmap='gray')
    fig.colorbar(mapp, ax=ax1)
    mapp = ax2.imshow(good_dphase, cmap='gray')
    fig.colorbar(mapp, ax=ax2)
    mapp = ax3.imshow(samantha_dphase-good_dphase, cmap='gray')
    fig.colorbar(mapp, ax=ax3)
    plt.show()

    #  inicial, final = 300,600
    #  im_ref = reference[inicial:final, inicial:final]
    #  im_def = deformed[inicial:final, inicial:final]

    #  mask_out = tukey_2d(im_ref.shape[0], 90, 30, 30)
    #  mask_in = tukey_2d(im_ref.shape[0], 90, 15, 30)

    #  #  good_dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(mask_in)
    #  frankestein = im_def*(mask_in) + im_ref*(1-mask_out)
    #  im_def_by_pieces = im_def*(mask_in) + im_def*(1-mask_out)

    #  im_ref = im_ref*(mask_in) + im_ref*(1-mask_out)
    #  dphase_mask = calculate_phase_diff_map_1D(frankestein, im_ref, th=0.9, ns=3)*(mask_in)

    #  good_dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(mask_in)

    #  good_dphase[mask_in != 1] = np.nan
    #  dphase_mask[mask_in != 1] = np.nan

    #  good_dphase -= np.nanmean(good_dphase)
    #  dphase_mask -= np.nanmean(dphase_mask)

    #  fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey= True)
    #  axes = axes.flatten()

    #  mapp = axes[0].imshow(good_dphase)
    #  axes[0].set_title('good dphase')
    #  fig.colorbar(mapp, ax=axes[0])

    #  mapp = axes[1].imshow(dphase_mask)
    #  axes[1].set_title('dphase mask')
    #  fig.colorbar(mapp, ax=axes[1])

    #  mapp =axes[2].imshow(good_dphase-dphase_mask)
    #  axes[2].set_title('dphase diff')
    #  fig.colorbar(mapp, ax=axes[2])

    #  mapp =axes[3].imshow(frankestein)
    #  axes[3].set_title('frankestein')
    #  fig.colorbar(mapp, ax=axes[3])

    #  mapp =axes[4].imshow(im_def_by_pieces)
    #  axes[4].set_title('im_def_by_pieces')
    #  fig.colorbar(mapp, ax=axes[4])

    #  axes[5].get_shared_y_axes().remove(axes[5])
    #  axes[5].set_ylim(-0.5, 1.5)
    #  axes[5].plot(mask_in[mask_in.shape[0]//2, :], label='mask_in')
    #  axes[5].plot(mask_out[mask_in.shape[0]//2, :], label='mask_out')
    #  axes[5].legend()

    #  fig.subplots_adjust(right=0.8)

    #  plt.show()

def test_annulus_final_med():
    from skimage.morphology import dilation, disk, binary_erosion
    import skimage.io as sio

    path = '../../../Mediciones_FaradayWaves/MED5 - No medimos - 0716/'
    gray = sio.imread(path+'gray/ID_0_C1S0001000001.tif').astype(float)
    deformed = sio.imread(path+'deformed/ID_0_C1S0001000001.tif').astype(float)-gray
    reference = sio.imread(path+'reference/ID_0_C1S0001000001.tif').astype(float)-gray

    annulus_mask, square_mask = np.load('MED5_masks.npy')
    x0, y0, r_inner, r_outer = np.load('MED5_circle_props.npy')
    x0 = int(x0)
    y0 = int(y0)

    mask_where_fringes_are = annulus_mask + square_mask

    full_def = _gerchberg2d(deformed, mask_where_fringes_are, 100)
    full_ref = _gerchberg2d(reference, mask_where_fringes_are, 100)

    good_dphase = calculate_phase_diff_map_1D(full_def, full_ref, th=0.9, ns=3, mask_for_unwrapping=1-annulus_mask)

    # Frankestain
    width = r_outer-r_inner
    r_middle = (r_outer+r_inner)//2
    mask_out = tukey_2d(x0, y0, 1024, r_middle, width*2, int(width))
    mask_in = tukey_2d(x0, y0, 1024, r_middle, width, int(width))

    frankestein = deformed*(mask_in) + full_ref*(1-mask_out)
    dphase_mask = calculate_phase_diff_map_1D(frankestein, full_ref, th=0.9, ns=3, mask_for_unwrapping=1-annulus_mask)

    for i in range(8): annulus_mask = binary_erosion(annulus_mask)
    good_dphase[annulus_mask == 0] = np.nan
    dphase_mask[annulus_mask == 0] = np.nan
    good_dphase -= np.nanmean(good_dphase)
    dphase_mask -= np.nanmean(dphase_mask)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey= True)
    axes = axes.flatten()

    mapp = axes[0].imshow(good_dphase)
    axes[0].set_title('good dphase')
    fig.colorbar(mapp, ax=axes[0])

    mapp = axes[1].imshow(dphase_mask)
    axes[1].set_title('dphase mask')
    fig.colorbar(mapp, ax=axes[1])

    mapp = axes[2].imshow(good_dphase-dphase_mask)
    axes[2].set_title('dphase diff')
    fig.colorbar(mapp, ax=axes[2])

    mapp = axes[3].imshow(frankestein)
    axes[3].set_title('frankestein')
    fig.colorbar(mapp, ax=axes[3])

    axes[4].get_shared_y_axes().remove(axes[4])
    axes[4].set_ylim(-0.5, 1.5)
    axes[4].plot(mask_in[mask_in.shape[0]//2, :], label='mask_in')
    axes[4].plot(mask_out[mask_in.shape[0]//2, :], label='mask_out')
    axes[4].legend()

    fig.subplots_adjust(right=0.8)

    plt.show()

    return

#  test_syntetic_phase()
#  test_square_img()
#  test_annulus_img()
test_annulus_final_med()
