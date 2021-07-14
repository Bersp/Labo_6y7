import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from unwrap import unwrap

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


        # OJO QUE ACA LO MODIFIQUE
        #gaussfilt1D= np.zeros([1,nx])
        gaussfilt1D= np.zeros(nx)
        gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)]=win

        # Multiplication by the filter
        Nfy0 = fY0*gaussfilt1D
        Nfy = fY*gaussfilt1D

        # Inverse Fourier transform of both images
        Ny0=np.fft.ifft(Nfy0)
        Ny=np.fft.ifft(Nfy)

        phase0[lin,:] = np.angle(Ny0)
        phase[lin,:]  = np.angle(Ny)

    # 2D-unwrapping is available with masks (as an option), using 'unwrap' library
    # unwrap allows for the use of wrapped_arrays, according to the docs:
    # "[...] in this case masked entries are ignored during the phase unwrapping process. This is useful if the wrapped phase data has holes or contains invalid entries. [...]"

    if mask_for_unwrapping is None:
        mphase0 = unwrap(phase0)
        mphase = unwrap(phase)
    else:
        mphase0 = ma.masked_array(phase0, mask=mask_for_unwrapping)
        mphase  = ma.masked_array(phase,  mask=mask_for_unwrapping)
        mphase0 = unwrap(mphase0)
        mphase = unwrap(mphase)

    # Definition of the phase difference map
    dphase = (mphase-mphase0);
    # dphase = dphase - np.min(dphase) - np.pi/2
    return dphase


@np.vectorize
def tukey_3d(x, radius, annulus_width, decay_width, x_len):

    x /= x_len

    r = annulus_width/(annulus_width+2*decay_width)
    xc = (x - radius)

    if xc >= 0 and xc < r/2:
        return 1/2*(1+np.cos(2*np.pi/r * (xc- r/2)))
    elif xc >= r/2 and xc < 1-r/2:
        return 1
    elif xc >= 1-r/2 and xc <= 1:
        return 1/2*(1+np.cos(2*np.pi/r * (xc-1 + r/2)))
    else:
        return 0

def tukey_2d(L, R, A, D):
    """
    construimos una ventana de tukey en 2D
    L = imagen resultante en tamaño L x L
    R = radio donde inicia la ventana de Tukey
    D = longitud de crecimiento/decrecimiento
    A = longitud del plateau
    """
    output = np.zeros((L,L))
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    r = np.sqrt( (x-L/2)**2 + (y-L/2)**2)
    region_plateau = (r>=(R-A/2)) * (r<=(R+A/2))
    region_subida  = (r>=(R-A/2-D)) * (r<(R-A/2))
    region_bajada  = (r>=(R+A/2)) * (r<(R+A/2+D))
    output[region_plateau] = 1
    output[region_subida]  = 0.5*(1-np.cos(np.pi/D*(r[region_subida]-np.mean(r[r==(R-A/2-D)]))))
    output[region_bajada]  = 0.5*(1+np.cos(np.pi/D*(r[region_bajada]-np.mean(r[r==(R+A/2)]))))
    r
    return output

def create_smooth_mask(radius, center, annulus_width, output_shape):
    x, y = [np.arange(0, output_shape)]*2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X-center[0])**2 + (Y-center[1])**2)

    sigma = annulus_width
    mask = 1 - np.exp( -(R-radius)**2/(2*sigma**2) )
    #  mask[mask > 0.8] = 1

    return mask

v = np.linspace(-1, 1, 1024)
x, y = np.meshgrid(v, v)
phase_imposed = (3*(1-x)**2.*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2))/2

f0 = 80
im_ref = np.sin(f0*x)
im_def = np.sin(f0*x + phase_imposed)

mask1 = tukey_2d(1024, 300, 50, 30)
mask2 = tukey_2d(1024, 300, 10, 30)

# El nuevo tukey las define al revés
mask1 = 1 - mask1
mask2 = 1- mask2
#  mask1 = create_smooth_mask(radius=450, center=(512, 512),
                          #  annulus_width=60, output_shape=1024)
#  mask2 = create_smooth_mask(radius=450, center=(512, 512),
                          #  annulus_width=10, output_shape=1024)

dphase = calculate_phase_diff_map_1D(im_def, im_ref, th=0.9, ns=3)*(1-mask2)
im_def1 = im_def*(1-mask2) + im_ref*mask1
im_def2 = im_def*(1-mask2) + im_def*mask1

im_ref = im_ref*(1-mask2) + im_ref*mask1
dphase_mask = calculate_phase_diff_map_1D(im_def1, im_ref, th=0.9, ns=3)*(1-mask2)

phase_imposed = phase_imposed*(1-mask2)
#  dphase = phase_imposed

dphase -= dphase[65, 458]
dphase_mask -= dphase_mask[65, 458]

dphase[np.isclose((1-mask2), 0, atol=0.4)] = np.nan
dphase_mask[np.isclose((1-mask2), 0, atol=0.4)] = np.nan

print(np.nanmax(dphase)-np.nanmin(dphase))
print(np.nanmax(dphase_mask)-np.nanmin(dphase_mask))

plt.imshow(im_def1)
plt.title('im_def1')
plt.colorbar()

plt.figure()
plt.imshow(im_def2)
plt.title('im_def2')
plt.colorbar()

plt.imshow(im_def1)
plt.title('def')
plt.colorbar()

plt.figure()
plt.imshow(dphase)
plt.title('dphase')
plt.colorbar()

plt.figure()
plt.imshow(dphase-dphase_mask)
plt.title('dphase diff')
plt.colorbar()

plt.figure()
plt.imshow(dphase_mask)
plt.title('dphase mask')
plt.colorbar()

plt.show()
