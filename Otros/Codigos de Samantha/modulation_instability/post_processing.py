import numpy as np
from unwrap import unwrap
from scipy import signal
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.signal import find_peaks, tukey

def pad_columns(array2d, pad_width=100):
    """
    Pad an array in a periodic way along the direction of the columns.

    Parameters
    ----------
    array2d : array
        Array to pad
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)

    Returns
    -------
    padded_array : array
        Padded array of rank equal to array with shape increased according to pad_width.
    """

    padded_array = np.zeros((array2d.shape[0],array2d.shape[1]+2*pad_width))

    padded_array[:,:pad_width]     = array2d[:,-pad_width:]
    padded_array[:,-pad_width:]    = array2d[:,:pad_width]

    padded_array[:,pad_width:-pad_width] = array2d

    return padded_array

def pad_lines(array2d, pad_width=100):
    """
    Pad an array in a period way along the direction of the lines.

    Parameters
    ----------
    array2d : array
        Array to pad
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)

    Returns
    -------
    padded_array : array
        Padded array of rank equal to array with shape increased according to pad_width.
    """

    padded_array = np.zeros((array2d.shape[0]+2*pad_width,array2d.shape[1]))

    padded_array[:pad_width,:]     = array2d[-pad_width:,:]
    padded_array[-pad_width:,:]    = array2d[:pad_width,:]

    padded_array[pad_width:-pad_width,:] = array2d

    return padded_array

def pad_total(array2d, pad_width=100):
    """
    Pad an array in both directions in a periodic way.

    Parameters
    ----------
    array2d : array
        Array to pad
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)

    Returns
    -------
    padded_array : array
        Padded array of rank equal to array with shape increased according to pad_width.
    """

    padded_columns = pad_columns(array2d, pad_width)
    padded_array   = pad_lines(padded_columns, pad_width)

    return padded_array

def calculate_phase_map(xt, ns=15, roll=True):
    """
    Obtains phase map from a spatiotemporal diagram.

    Parameters
    ----------
    xt : array
        Spatiotemporal diagram
    ns : int, optional
        Size of the gaussian filter (default is 15)
    roll: bool, optional
        If True, centers the signal before perfoeming the inverse Fourier Transform to obtain the phase

    Returns
    -------
    phase : array
        Phase map
    """

    phase = np.zeros_like(xt, dtype='float')
    ft    = np.fft.fft(xt, axis=1)

    ny, nx = np.shape(ft)
    th     = 0.9
    num    = 9

    max_array = np.zeros(xt.shape[0])
    for i in range(xt.shape[0]):
        ftlin = ft[i,:]
        imax=np.argmax(np.abs(ftlin[num:int(np.floor(nx/2))]))+num
        max_array[i] = imax
    ifmax = int(np.mean(max_array))

    HW=np.round(ifmax*th)
    W=2*HW
    win=signal.tukey(int(W),ns)

    gaussfilt1D= np.zeros(nx)
    gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)]=win

    for i in range(xt.shape[0]):
        ftlin = ft[i,:]
        # Multiplication by the filter
        Nfy = ftlin*gaussfilt1D
        if roll==True:
            c_centered = np.roll(Nfy, -ifmax)
        else:
            c_centered = Nfy
        # Inverse Fourier transform
        Cxt = np.fft.ifft(c_centered)
        # 1D unwrapping
        phi = np.unwrap(np.imag(np.log(Cxt)))
        phase[i,:] = phi

    # 2D unwrapping
    phase = unwrap(phase)

    # Mean substraction
    for i in range(xt.shape[0]):
        line_phase = np.unwrap(phase[i,:])
        phase[i,:] = line_phase-np.mean(line_phase)

    return phase



def check_lines(phase_selection, time_idx, pad_width=100):
    """
    Check if the lines used to build the phase selection are "good" (smooth) lines

    Parameters
    ----------
    phase_selection : array
        Selected lines from phase map
    time_idx : array
        Indexes of the time array corresponding to the selected lines
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)

    Returns
    -------
    phase_selection : array
        Selected lines from phase map
    time_idx : array
        Indexes of the time array corresponding to the selected lines
    """

    check=True
    while check==True:
        p0 = phase_selection[:,0]

        peaks_p0_pos, _        = signal.find_peaks(p0, height = np.mean(p0))
        threshold_pos          = np.mean(p0[peaks_p0_pos])
        idxs_to_check_line_pos = np.argwhere(p0>threshold_pos)

        peaks_p0_neg, _        = signal.find_peaks(-p0, height = np.mean(-p0))
        threshold_neg          = np.mean(-p0[peaks_p0_neg])+np.std(-p0[peaks_p0_neg])
        idxs_to_check_line_neg = np.argwhere(-p0>threshold_neg)

        d0 = np.abs(np.diff(phase_selection[:,0]))

        idxs_to_check_diff = np.argwhere(d0>(np.mean(d0)+3*np.std(d0)))+1
        idxs_to_check_pos  = np.intersect1d(idxs_to_check_line_pos, idxs_to_check_diff)

        idxs_to_check_neg  = np.intersect1d(idxs_to_check_line_neg, idxs_to_check_diff)
        idxs_to_check      = np.concatenate((idxs_to_check_pos, idxs_to_check_neg))

        idxs_to_check = np.delete(idxs_to_check, np.argwhere(idxs_to_check>(pad_width-5)))
        idxs_to_check = np.delete(idxs_to_check, np.argwhere(idxs_to_check<(pad_width+5)))

        if len(idxs_to_check)>0:
            check=True
            for i in range(len(idxs_to_check)):
                idx_to_check = idxs_to_check[i]
                t_to_check = tiempos_idx[idx_to_check]

                # Replace the current line with the previous or the next one
                direccion_movimiento = np.argmax((np.max(xt_padded_total[t_to_check-1,:]), np.max(xt_padded_total[t_to_check+1,:])))

                if direccion_movimiento==0:
                # Replace with the previous line
                    linea_fase_to_check =  np.unwrap(fase_unwrap[t_to_check-1,:])
                    fase_periodo[idx_to_check,:] = linea_fase_to_check - np.mean(linea_fase_to_check)
                    tiempos_idx[idx_to_check] = t_to_check-1

                if direccion_movimiento==1:
                # Replace with the next line
                    linea_fase_to_check =  np.unwrap(fase_unwrap[t_to_check+1,:])
                    fase_periodo[idx_to_check,:] = linea_fase_to_check - np.mean(linea_fase_to_check)
                    tiempos_idx[idx_to_check] = t_to_check+1

        if len(idxs_to_check)==0:
            check=False

    return phase_selection, time_idx



def calculate_padded_envelope(xt, dx, dt, pad_width=100, dist=100):
    """
    Calcula la envolvente de array 2D en la direccion de columnas y devuelve un array paddeado.

    Parameters
    ----------
    xt : array
        Spatiotemporal diagram
    dx : float
        Spatial step
    dt : float
        Temporal step
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)

    Returns
    -------
    envelope_padded : array
        Interpolated envelope
    """
    xt_padded = pad_total(xt)
    t_padded  = np.arange(0, xt_padded.shape[0]*dt, dt)
    x_padded  = np.arange(0, xt_padded.shape[1]*dx, dx)[:xt.shape[1]+2*pad_width]

    envelope_padded = np.zeros_like(xt_padded, dtype='float')
    for i in range(len(t_padded)):
        line = xt_padded[i,:]

        peaks, _ = find_peaks(line, height=np.mean(line), distance=dist)

        idx_cut = np.argwhere(line[peaks]<0)
        line[peaks[idx_cut]] = 0

        spline_line = CubicSpline(peaks,line[peaks])
        line_envelope = spline_line(np.arange(len(line)))

        envelope_padded[i,:] = line_envelope

    return envelope_padded

def calculate_envelope(xt, dx, dt, pad_width=100, dist=100, interpolate=True, **kwargs):
    """
    Calcula la envolvente de array 2D en la direccion de columnas e interpola tomando una linea por periodo.

    Parameters
    ----------
    xt : array
        Spatiotemporal diagrm
    dx : float
        Spatial step
    dt : float
        Temporal step
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)
    dist : int, optional
        Distance between peaks used to find the envelope for a fixed time
    interpolate: bool, optional
        If True, keeps only one line per period (default is True)
    pad_width : int, optional
        Number of values padded to the edges of each axis (default is 100)
    column_idx : int, optional
        Index of the column used to detemine the lines to keep
    temporal_step : int, optional
        Frames per period

    Returns
    -------
    envelope : array
        Interpolated envelope
    dt_envelope : array
        First temporal derivative of the envelope
    dx_envelope : array
        First spatial derivative of the envelope
    dx2_envelope : array
        Second spatial derivative of the envelope
    """

    xt_padded = pad_total(xt, pad_width)
    t_padded  = np.arange(0, xt_padded.shape[0]*dt, dt)
    x_padded  = np.arange(0, xt_padded.shape[1]*dx, dx)[:xt.shape[1]+2*pad_width]

    # Envelope
    prev_envelope = calculate_padded_envelope(xt, dx, dt, pad_width, dist)
    if interpolate==True:
        # Keep one line per period
        if 'time_idx' in kwargs:
            time_idx = kwargs['time_idx']
        else:
            if 'temporal_step' in kwargs:
                temporal_step = kwargs['temporal_step']
                time_idx = np.arange(0, xt.shape[0], temporal_step)
            else:
                if 'column_idx' in kwargs:
                    column_idx = kwargs['column_idx']
                else:
                    column_idx = (xt.shape[1] + 2*pad_width)//2

                column    = xt_padded[:, column_idx]
                time_idx, _ = signal.find_peaks(column, height=1.1*np.mean(column), distance = 12)

        envelope_selection = np.zeros((len(time_idx), xt_padded.shape[1]))
        for i in range(len(time_idx)):
            ti   = time_idx[i]
            line = prev_envelope[ti,:]
            envelope_selection[i,:] = line
    else:
       envelope_selection = prev_envelope

    # Spline interpolation and derivatives calculation
    if interpolate==True:
        spline_envelope = RectBivariateSpline(t_padded[time_idx],x_padded, envelope_selection)
    else:
        spline_envelope = RectBivariateSpline(t_padded,x_padded, envelope_selection)
    envelope_padded = spline_envelope(t_padded,x_padded)
    dt_envelope_padded  = spline_envelope(t_padded, x_padded, dx=1, dy=0)
    dx_envelope_padded = spline_envelope(t_padded, x_padded, dx=0, dy=1)
    dx2_envelope_padded = spline_envelope(t_padded, x_padded, dx=0, dy=2)

    # Cut the padded values
    envelope = envelope_padded[pad_width:-pad_width,pad_width:-pad_width]
    dt_envelope  = dt_envelope_padded[pad_width:-pad_width,pad_width:-pad_width]
    dx2_envelope = dx2_envelope_padded[pad_width:-pad_width,pad_width:-pad_width]
    dx_envelope  = dx_envelope_padded[pad_width:-pad_width,pad_width:-pad_width]

    return envelope, dt_envelope, dx_envelope, dx2_envelope


def calculate_phase(xt, dx, dt, ns=15, interpolate=True, check=False, keep_phase_perturbation=False, roll=False,  pad_width=100, **kwargs):
    """
    This function calculates the phase map from a spatiotemporal diagram.

    Parameters
    ----------
    xt : array
        Spatiotemporal diagram
    xfinal : float
        Lenght of the 1D region considered
    dx : float
        Spatial step
    dt : float
        Temporal step
    temporal_step : int, optional
        Frames per period
    ns : int, optional
        Size of the gaussian filter (default is 15)
        Distance between peaks used to find the envelope for a fixed time
    interpolate: bool, optional
        If True, keeps only one line per period (default is True)
    check : bool, optional
        If True, checks that the lines used to build the phase map be "good" (smooth) lines (default is False)
    keep_phase_perturbation :  bool, optional
        If True, we only keep the phase pertirbation (whitout the 'kx' component) (default is False)
    roll: bool, optional
        If True, centers the signal before perfoeming the inverse Fourier Transform to obtain the phase
    column_idx : int, optional
        Index of the column used to detemine the lines to keep
    temporal_step : int, optional
        Frames per period

    Returns
    -------
    phase : array
        Interpolated phase map
    time_idx : array
        Indexes of the time array corresponding to the selected lines
    dt_phase : array
        First temporal derivative of the phase map
    dx_phase : array
        First spatial derivative of the phase map
    dx2_phase : array
        Second spatial derivative of the phase map
    """

    # Phase map
    phase = calculate_phase_map(xt, ns, roll=roll)
    x     = np.arange(0, xt.shape[1]*dx, dx)
    t     = np.arange(0, xt.shape[0]*dt, dt)

    if interpolate==True:
        # Keep one line per period
        if 'time_idx' in kwargs:
            time_idx = kwargs['time_idx']
        else:
            if 'temporal_step' in kwargs:
                temporal_step = kwargs['temporal_step']
                time_idx = np.arange(0, xt.shape[0], temporal_step)
            else:
                if 'column_idx' in kwargs:
                    column_idx = kwargs['column_idx']
                else:
                    xt_padded   = pad_total(xt, pad_width)
                    column_idx  = (xt.shape[1] + 2*pad_width)//2
                    column      = xt_padded[:, column_idx]
                    column      = column[pad_width:-pad_width]
                    time_idx, _ = signal.find_peaks(column, height=1.1*np.mean(column), distance = 12)

        phase_selection = np.zeros((len(time_idx), len(x)))
        for i in range(len(time_idx)):
            ti = time_idx[i]
            phase_selection[i,:] = phase[ti,:]

        if check==True:
            phase_selection, time_idx = check_lines(phase_selection, time_idx, pad_width)
    else:
        phase_selection=phase
        time_idx = []


    # Phase difference
    phase_diff = np.zeros_like(phase_selection, dtype=float)
    for i in range(phase_selection.shape[0]):
        m, b = np.polyfit(x, phase_selection[i,:], 1)
        phase_diff[i,:] = phase_selection[i,:] - (m*x+b)

    # Spline interpolation and derivatives calculation
    if interpolate==True:
        if keep_phase_perturbation==True:
            spline_phase     = RectBivariateSpline(t[time_idx],x, phase_diff)
        else:
            spline_phase     = RectBivariateSpline(t[time_idx],x, phase_selection)
    if interpolate==False:
        if keep_phase_perturbation==True:
            spline_phase     = RectBivariateSpline(t,x, phase_diff)
        else:
            spline_phase     = RectBivariateSpline(t,x, phase_selection)
    phase     = spline_phase(t,x)
    dt_phase  = spline_phase(t, x, dx=1, dy=0)
    dx_phase  = spline_phase(t, x, dx=0, dy=1)
    dx2_phase = spline_phase(t, x, dx=0, dy=2)

    return phase, dt_phase, dx_phase, dx2_phase, time_idx

# def calculate_phase_difference(phase, xfinal, dx, pad_width=100):
#     """
#     Subtraction between phase map and a linear function, in order to preserve the periodic boundary conditions.
# 
#     Parameters
#     ----------
#     phase : array
#         Phase map
#     xfinal : float
#         Lenght of the 1D region considered
#     dx : float
#         Spatial step
#     pad_width : int, optional
#         Number of values padded to the edges of each axis (default is 100)
# 
#     Returns
#     -------
#     phase_diff : array
#         Phase difference between phase map and the linear function
#     """
# 
#     x_padded   = np.arange(0, phase.shape[1]*dx, dx)[:phase.shape[1]]
#     phase_diff = np.zeros_like(phase, dtype=float)
# 
#     for i in range(phase.shape[0]):
#         slope = (phase[i,-pad_width]-phase[i,pad_width])/xfinal
#         phase_diff[i,:] = phase[i,:] - slope*x_padded
# 
#     return phase_diff
