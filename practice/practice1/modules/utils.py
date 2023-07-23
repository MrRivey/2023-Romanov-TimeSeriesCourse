import numpy as np
import math


def z_normalize(ts):

    """
    Calculate the z-normalized time series by subtracting the mean and
    dividing by the standard deviation along a given axis.

    Parameters
    ----------
    ts : numpy.ndarray
        Time series.
    
    Returns
    -------
    norm_ts : numpy.ndarray
        The z-normalized time series.
    """

    norm_ts = (ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)

    return norm_ts



def sliding_window(ts, window, step=1):
    """
    Extract subsequences from time series using sliding window.

    Parameters
    ----------
    ts : numpy.ndarray
        Time series.

    window : int
        Size of the sliding window.

    step : int
        Step of the sliding window.

    Returns
    -------
    subs_matrix : numpy.ndarray
        Matrix of subsequences.
    """
    
    n = ts.shape[0]
    N = math.ceil((n-window+1)/step)

    subs_matrix = np.zeros((N, window))

    for i in range(N):
        start_idx = i*step
        end_idx = start_idx + window
        subs_matrix[i] = ts[start_idx:end_idx]

    return subs_matrix