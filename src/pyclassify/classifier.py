from . import utils
import numpy as np
class kNN :
    def __init__(self, k_init, backend_init) :
        if not(isinstance(k_init, int)) :
            raise TypeError ("k must be integer")
        if backend_init != "plain" and backend_init != "numpy" :
            raise ValueError ( "backend must be either \"plain\" or \"numpy\"")
        self.k = k_init
        self.backend = backend_init

    @profile
    def _get_k_nearest_neighbours(self,X,y,x) :
        distances = []
        for i in range(len(X)) :
            distances.append(self.distance(X[i],x))
        d_y = list(zip(distances,y))
        d_y.sort()
        y_sorted = [item[1] for item in d_y]
        return y_sorted[:self.k]

    @profile
    def __call__(self, data, new_points):
        X, y = data
        retval = []
        if self.backend == "plain" :
            self.distance = utils.distance
        else :
            self.distance = utils.distance_numpy
            X, y = np.array(X),  np.array(y)
            
        for new_point in new_points :
            neighbours_labels = self._get_k_nearest_neighbours(X,y,new_point)
            retval.append(utils.majority_vote(neighbours_labels))
        return retval