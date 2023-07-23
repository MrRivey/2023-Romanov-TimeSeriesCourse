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

        N, m = self.ts_data.shape
        bsf = float("inf")

        if (self.excl_zone_denom is None):
            excl_zone = 0
        else:
            excl_zone = int(np.ceil(m / self.excl_zone_denom))

        if (self.normalize):
            query = z_normalize(self.query)

        distances = []

        for i in range(N):

            subsequence = self.ts_data[i]
            if (self.normalize):
                subsequence = z_normalize(subsequence)

            dist = DTW_distance(subsequence, query, self.r)

            if (bsf < dist):
                distances.append(np.inf)
            else:
                distances.append(dist)
                self.bestmatch = super()._top_k_match(distances, m, bsf, excl_zone)
                if (len(self.bestmatch['index']) == self.top_k):
                    bsf = self.bestmatch['distance'][self.top_k-1]

            #print(self.bestmatch)

        return self.bestmatch


class UCR_DTW(BestMatchFinder):

    def __init__(self, ts=None, query=None, exclusion_zone=None, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)


    def _LB_Kim(self, subs1, subs2):

        lb_Kim = 0
        lb_Kim = (np.max(subs1) - np.max(subs2))**2 + (np.min(subs1) - np.min(subs2))**2

        return lb_Kim


    def _LB_Keogh(self, subs1, subs2, r):

        lb_Keogh = 0

        r = int(np.floor(r * len(subs1)))

        for i in range(len(subs1)):

            start_idx = max(0, i - r)
            stop_idx = min(i + r, len(subs1))

            if (start_idx != stop_idx):
                lower_bound = min(subs2[start_idx:stop_idx])
                upper_bound = max(subs2[start_idx:stop_idx])
            else:
                lower_bound = subs2[i]
                upper_bound = subs2[i]

            if subs1[i] > upper_bound:
                lb_Keogh = lb_Keogh + (subs1[i] - upper_bound)**2
            elif subs1[i] < lower_bound:
                lb_Keogh = lb_Keogh + (subs1[i] - lower_bound)**2

        return lb_Keogh


    def perform(self):

        N, m = self.ts_data.shape

        bsf = float("inf")

        if (self.excl_zone_denom is None):
            excl_zone = 0
        else:
            excl_zone = int(np.ceil(m / self.excl_zone_denom))

        if (self.normalize):
            query = z_normalize(self.query)

        distances = []

        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

        for i in range(N):

            subsequence = self.ts_data[i]

            if (self.normalize):
                subsequence = z_normalize(subsequence)

            pruned = False

            lb_Kim = self._LB_Kim(subsequence, query)

            if (lb_Kim < bsf):

                lb_KeoghQC = self._LB_Keogh(query, subsequence, self.r)

                if (lb_KeoghQC < bsf):

                    lb_KeoghCQ = self._LB_Keogh(subsequence, query, self.r)

                    if (lb_KeoghCQ < bsf):

                        dist = DTW_distance(subsequence, query, self.r)

                        if (dist < bsf):
                            distances.append(dist)
                            self.bestmatch = super()._top_k_match(distances, m, bsf, excl_zone)

                            if (len(self.bestmatch['index']) == self.top_k):
                                bsf = self.bestmatch['distance'][self.top_k-1]
                                #print(f'bsf = {bsf}')

                        else:
                            pruned = True
                    else:
                        self.lb_KeoghCQ_num = self.lb_KeoghCQ_num + 1
                        pruned = True
                else:
                    self.lb_KeoghQC_num = self.lb_KeoghQC_num + 1
                    pruned = True
            else:
                self.lb_Kim_num = self.lb_Kim_num + 1
                pruned = True

            if (pruned):
                distances.append(np.inf)

            #print(self.bestmatch)

        return {'topk_match': self.bestmatch,
                'lb_Kim_num': self.lb_Kim_num,
                'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
                'lb_KeoghQC_num': self.lb_KeoghQC_num
                }