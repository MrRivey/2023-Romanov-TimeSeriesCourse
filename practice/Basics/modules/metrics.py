import numpy as np


def ED_distance(ts1, ts2):

    square = np.square(ts1 - ts2)
    sum_square = np.sum(square)
    ed_dist = np.sqrt(sum_square)
    
    return ed_dist


def norm_ED_distance(ts1, ts2):
  
    norm_ed_dist = 0

    avg_ts1 = np.mean(ts1)
    avg_ts2 = np.mean(ts2)

    std_ts1 = np.std(ts1)
    std_ts2 = np.std(ts2)

    T1T2 = ts1.dot(ts2)
    drob = (T1T2-avg_ts1*avg_ts2*len(ts1))/(std_ts1*std_ts2*len(ts1))

    norm_ed_dist = abs(2*len(ts1)*(1-drob))**0.5
    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    
    m = sum(ts1.shape)
    d = np.zeros((m,m))
    for i in range(m):
      for j in range(m):
        d[i,j] = (ts1[i]-ts2[j])**2

    D = np.zeros((m+1,m+1))

    for i in range(1, m + 1):
      D[i,0] = np.inf
    for i in range(1, m + 1):
      D[0,i] = np.inf
    for i in range(1, m+1):
      for j in range(1, m + 1):
        D[i, j] = d[i - 1][j - 1] + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
        


    dtw_dist = D[m][m]
    return dtw_dist