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

    # INSERT YOUR CODE

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
