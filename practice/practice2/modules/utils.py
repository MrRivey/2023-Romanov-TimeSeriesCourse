import numpy as np
import math
import random


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



def random_walk(n):
    """
    Generate the time series based on Random Walk model.

    Parameters
    ----------
    n : int
        The length of time series.
    
    Returns
    -------
    random_walk_ts : numpy.ndarray
        The generated time series.

    """

    value = 0

    random_walk_ts = [value]
    directions = ["UP", "DOWN"]

    for i in range(1, n):
        # Randomly select either UP or DOWN
        step = random.choice(directions)

        # Move the object up or down
        if step == "UP":
            value += 1
        elif step == "DOWN":
            value -= 1

        random_walk_ts.append(value)

    return np.array(random_walk_ts)


