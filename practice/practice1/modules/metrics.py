import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0

    for i in range(len(ts1)):
        ed_dist += (ts1[i] - ts2[i])**2

    ed_dist = np.sqrt(ed_dist)

    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    n = len(ts1)

    mean1 = np.mean(ts1, axis=0)
    mean2 = np.mean(ts2, axis=0)

    std1 = np.std(ts1, axis=0)
    std2 = np.std(ts2, axis=0)

    dot = np.dot(ts1, ts2)

    norm_ed_dist = np.sqrt(2*n*(1-(dot-n*mean1*mean2)/(n*std1*std2)))

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """


    len1 = len(ts1)
    len2 = len(ts2)

    if (r is None):
        r = 1
    else:
        if ((r > 1) or (r < 0)):
            raise ValueError('The parameter r is wrong. It should be between 0 and 1.')

    r = int(np.floor(r * len1))

    cost = np.zeros(shape=(len1+1, len2+1))


    for i in range(len1+1):
        for j in range(len2+1):
            cost[i, j] = float("inf")

    cost[0, 0] = 0

    for i in range(1, len1+1):
        for j in range(max(1, i-r), min(len2, i+r)+1):
            cost[i, j] = 0

    for i in range(1, len1+1):

        start_j = max(1, i-r)
        end_j = min(len2, i+r)+1

        for j in range(start_j, end_j):
            dist = (ts1[i-1] - ts2[j-1])**2
            cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    dtw_dist = cost[len1, len2]

    return dtw_dist