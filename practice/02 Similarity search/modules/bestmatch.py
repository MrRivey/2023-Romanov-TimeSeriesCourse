import numpy as np
import copy

from modules.utils import *
from modules.metrics import *


class BestMatchFinder:

    def __init__(self, ts_data=None, query=None, exclusion_zone=None, top_k=3, normalize=True, r=0.05):

        self.query = copy.deepcopy(np.array(query))
        if (len(ts_data.shape) == 2): # time series set
            self.ts_data = ts_data
        else:
            self.ts_data = sliding_window(ts_data, len(query))

        self.excl_zone_denom = exclusion_zone
        self.top_k = top_k
        self.normalize = normalize
        self.r = r


    def _apply_exclusion_zone(self, a, idx, excl_zone):

        zone_start = max(0, idx - excl_zone)
        zone_stop = min(a.shape[-1], idx + excl_zone)
        a[zone_start : zone_stop + 1] = np.inf

        return a


    def _top_k_match(self, distances, m, bsf, excl_zone):

        data_len = len(distances)
        top_k_match = []

        distances = np.copy(distances)
        top_k_match_idx = []
        top_k_match_dist = []

        for i in range(self.top_k):
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > bsf):
                break

            distances = self._apply_exclusion_zone(distances, min_idx, excl_zone)

            top_k_match_idx.append(min_idx)
            top_k_match_dist.append(min_dist)

        return {'index': top_k_match_idx, 'distance': top_k_match_dist}


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):

    def __init__(self, ts=None, query=None, exclusion_zone=1, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)


    def perform(self):

        # INSERT YOUR CODE

        return self.bestmatch


class UCR_DTW(BestMatchFinder):

    def __init__(self, ts=None, query=None, exclusion_zone=None, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)


    def _LB_Kim(self, subs1, subs2):

        lb_Kim = 0

        # INSERT YOUR CODE
        
        return lb_Kim


    def _LB_Keogh(self, subs1, subs2, r):

        lb_Keogh = 0

        # INSERT YOUR CODE

        return lb_Keogh


    def perform(self):

        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0
        
        # INSERT YOUR CODE

        return {'index' : self.bestmatch['index'],
                'distance' : self.bestmatch['distance'],
                'lb_Kim_num': self.lb_Kim_num,
                'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
                'lb_KeoghQC_num': self.lb_KeoghQC_num
                }