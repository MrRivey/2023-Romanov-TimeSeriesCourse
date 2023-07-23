import numpy as np

from modules.utils import *

def top_k_motifs(matrix_profile, top_k=3):
    """
    Find the top-k motifs based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of motifs.

    Returns
    --------
    motifs : dict
        Top-k motifs (left and right indices and distances).
    """

    motifs_idx = []
    motifs_dist = []

    mp = np.copy(matrix_profile['mp']).astype(np.float64)
    mpi = matrix_profile['mpi']

    exclusion_zone = matrix_profile['excl_zone']

    for i in range(top_k):
        min_idx = np.argmin(mp)
        min_dist = mp[min_idx]

        if is_nan_inf(min_dist):
            break

        first_motif_idx = np.min([min_idx, mpi[min_idx]])
        second_motif_idx = np.max([min_idx, mpi[min_idx]])

        for j in (first_motif_idx, second_motif_idx):
            mp = apply_exclusion_zone(mp, j, exclusion_zone, np.inf)
        
        motifs_idx.append((first_motif_idx, second_motif_idx))
        motifs_dist.append(min_dist)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }