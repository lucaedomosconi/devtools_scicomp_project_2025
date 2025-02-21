from . import utils
from . import utils_compiled
import numpy as np
from line_profiler import profile

class kNN :
    """
    kNN Classifier
    This class implements a k-Nearest Neighbors (kNN) classifier with support for 
    multiple backends for distance computation: "plain", "numpy", and "numba".
    Attributes:
    k : int
        The number of nearest neighbors to consider for classification.
    backend : str
        The backend to use for distance computation. Must be one of "plain", "numpy", or "numba".
    Methods:
    __init__(k, backend)
        Initializes the kNN classifier with the specified number of neighbors and backend.
    _get_k_nearest_neighbours(X, y, x)
        Finds the k-nearest neighbors of a given data point.
    __call__(data, new_points)
        Classifies new data points based on the provided training data.
    """
    def __init__(self, k_init, backend_init) :
        if not(isinstance(k_init, int)) :
            raise TypeError ("k must be integer")
        if not(isinstance(backend_init, str)) :
            raise TypeError ("backend must be a string type")
        if backend_init != "plain" and backend_init != "numpy" and backend_init != "numba" :
            raise ValueError ( "backend must be either \"plain\", \"numpy\" or \"numba\"")
        self.k = k_init
        self.backend = backend_init

    @profile
    def _get_k_nearest_neighbours(self,X,y,x) :
        """
        Find the k-nearest neighbours of a given point x.

        Parameters:
        -----------
        X : list of lists of floats or a 2D NumPy array.
            The training data points.
        y : list of strings or a 1D NumPy array.
            The labels corresponding to the training data points.
        x : list of floats or a 1D NumPy array.
            The data point for which to find the k-nearest neighbours.

        Returns:
        --------
        list
            A list of the labels of the k-nearest neighbours.
        """
        distances = [self.distance(X[i],x) for i in range(len(X))]
        return [item[1] for item in sorted(list(zip(distances,y)))][:self.k]

    @profile
    def __call__(self, data : tuple[list[list[float]],list], new_points : list[list[float]]) -> list :
        """
        Classify new data points based on the provided training data.

        Parameters:
        -----------
        data : tuple[list[list[float]], list]
            A tuple containing the training data. The first element is a list of feature vectors (X),
            and the second element is a list of corresponding labels (y).
        new_points : list[list[float]]
            A list of new data points to classify.

        Returns:
        --------
        list
            A list of predicted labels for each new data point.

        Notes:
        ------
        The method uses different backends ("plain", "numpy", "numba") to compute distances
        between points. The backend is determined by the `self.backend` attribute.
        """
        X, y = data
        retval = []
        if self.backend == "plain" :
            print("Using plain")
            self.distance = utils.distance
        elif self.backend == "numpy" :
            print("Using numpy")
            self.distance = utils.distance_numpy
            X, y, new_points = np.array(X),  np.array(y), np.array(new_points)
        elif self.backend == "numba" :
            print("Using numba")
            self.distance = utils_compiled.distance_numba
            X, y, new_points = np.array(X,dtype=np.float64),  np.array(y), np.array(new_points, np.dtype(np.float64))
        for new_point in new_points :
            neighbours_labels = self._get_k_nearest_neighbours(X,y,new_point)
            retval.append(utils.majority_vote(neighbours_labels))
        return retval