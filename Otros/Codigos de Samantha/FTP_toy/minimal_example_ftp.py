import numpy as np
from matplotlib import pyplot as plt
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

# Levanto las imagenes
gray       = np.load('gray.npy')
reference  = np.load('reference.npy')
deformed   = np.load('deformed.npy')


# Recorto un cuadrado en el medio
inicial, final = 300,600

gray = gray[inicial:final, inicial:final]
reference = reference[inicial:final, inicial:final]
deformed = deformed[inicial:final, inicial:final]


# Resto la gris
ref_m_gray = reference - gray
def_m_gray = deformed - gray

# Calculo la diferencia de fase entre la deformada y la de referencia
dphase = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, th=0.9, ns=3)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.imshow(deformed, cmap="gray")
ax1.set_title('deformed')
ax2.imshow(dphase, cmap="gray")
plt.show()
