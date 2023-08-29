import numpy as np


def is_nan_inf(val):
    """
    Check if the array contains np.nan, -np.nan, or np.inf values.

    Parameters
    ----------
    a : numpy.ndarray
        Array.

    Returns
    -------
    output : bool.
    """

    return np.isnan(val) or np.isinf(abs(val))


def apply_exclusion_zone(a, idx, excl_zone, val):
    """ 
    Set all values of array to `val` in a window around a given index.  

    Parameters
    ----------
    a : numpy.ndarray
        Array.

    idx : int
        The index around which the window should be centered.

    excl_zone : int
        Size of the exclusion zone.

    val : float
        The elements within the exclusion zone will be set to this value.

    Returns
    -------
    a : numpy.ndarray
        Array that is applied an exclusion zone.
    """
    
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)

    a[zone_start : zone_stop + 1] = val

    return a