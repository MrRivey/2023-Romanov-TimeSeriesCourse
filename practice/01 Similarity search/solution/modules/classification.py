import numpy as np

from modules.metrics import *
from modules.utils import z_normalize
from modules.bestmatch import *


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05, 'use_lb': True}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier.

    Parameters
    ----------
    n_neighbors : int, default = 3
        Number of neighbors.
    
    metric : str, default = 'euclidean'
        Distance measure between time series.
        Options: {euclidean, dtw}.

    metric_params : dict, default = None
        Dictionary containing parameters for the distance metric being used.
    """
    
    def __init__(self, n_neighbors=3, metric='euclidean', metric_params=None):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)


    def fit(self, X_train, Y_train):
        """
        Fit the model using X_train as training data and Y_train as labels.

        Parameters
        ----------
        X_train : numpy.ndarrray (2d array of shape (ts_number, ts_length))
            The train set.
        
        Y_train : numpy.ndarrray
            Labels of the train set.
        """
       
        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()


    def _distance(self, x_train, x_test):
        """
        Compute distance between the train and test samples.
        
        Parameters
        ----------
        x_train : numpy.ndarrray
            Sample of the train set.
        
        x_test : numpy.ndarrray
            Sample of the test set.
        
        Returns
        -------
        dist : float
            The distance between the train and test samples.
        """

        dist = 0

        if (self.metric == 'euclidean'):
            if (self.metric_params['normalize']):
                dist = norm_ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)
        else:
            if (self.metric == 'dtw'):
                if (self.metric_params['normalize']):
                    x_train = z_normalize(x_train)
                    x_test = z_normalize(x_test)
                dist = DTW_distance(x_train, x_test, r=self.metric_params['r'])

        return dist


    def _find_neighbors(self, x_test):
        """
        Find the k nearest neighbors of the test sample.

        Parameters
        ----------
        x_test : numpy.ndarray
            Sample of the test set.
        
        Returns
        -------
        neighbors : list of tuples (float, int)
            k nearest neighbors (distance between neighbor and test sample, neighbor label) for test sample.
        """

        distances = []
        neighbors = []

        if (self.metric == 'dtw') and (self.metric_params['use_lb']):

            ucr_dtw_model = UCR_DTW(self.X_train, x_test, None, self.n_neighbors, self.metric_params['normalize'], self.metric_params['r'])
            best_matches = ucr_dtw_model.perform()

            for i in range(self.n_neighbors):
                dist = best_matches['topk_match']['distance'][i]
                idx = best_matches['topk_match']['index'][i]
                neighbors.append((dist, self.Y_train[idx]))
        else:
            for i in range(len(self.X_train)):

                dist = self._distance(self.X_train[i], x_test)
                distances.append((dist, self.Y_train[i]))

            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.n_neighbors]

        return neighbors


    def predict(self, X_test):
        """
        Predict the class labels for samples of the test set.

        Parameters
        ----------
        X_test : numpy.ndarray (2d array of shape (ts_number, ts_length))
            The test set.

        Returns
        -------
        y_pred : list
            Class labels for each data sample from test set.
        """

        y_pred = []

        X_test = X_test.to_numpy()

        for x_test in X_test:

            neighbors = self._find_neighbors(x_test)
            neighbors_labels = (list(zip(*neighbors))[1])
            y_pred_i = max(set(neighbors_labels), key = neighbors_labels.count)
            y_pred.append(y_pred_i)

        return y_pred


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy classification score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth (correct) labels.

    y_pred : numpy.ndarray
        Predicted labels returned by a classifier.

    Returns
    -------
    score : float
        Accuracy classification score.
    """

    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score