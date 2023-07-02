import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile, top_k=3):
    """
    Find the top-k discords based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of discords.

    Returns
    --------
    discords : dict
        Top-k discords (indices, distances to its nearest neighbor 
        and the nearest neighbors indices).
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = np.copy(matrix_profile['mp']).astype(np.float64)
    mpi = matrix_profile['mpi']

    exclusion_zone = matrix_profile['excl_zone']
    
    for i in range(top_k):
        discord_idx = np.argmax(mp)
        discord_dist = mp[discord_idx]
        nn_idx = mpi[discord_idx]

        if is_nan_inf(discord_dist):
            break

        mp = apply_exclusion_zone(mp, discord_idx, exclusion_zone, val=-np.inf)

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }