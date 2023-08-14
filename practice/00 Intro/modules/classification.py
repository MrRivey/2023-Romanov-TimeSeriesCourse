import numpy as np

from modules.metrics import *
from modules.utils import z_normalize


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
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

        # INSERT YOUR CODE

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

        neighbors = []
        
        # INSERT YOUR CODE

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

        # INSERT YOUR CODE

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