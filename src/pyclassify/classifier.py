from . import utils
class kNN :
    def __init__(self, k_init) :
        if not(isinstance(k_init, int)) :
            raise TypeError ("input must be integer")
        self.k = k_init
    
    def _get_k_nearest_neighbours(self,X,y,x) :
        distances = []
        for i in range(len(X)) :
            distances.append(utils.distance(X[i],x))
        d_y = list(zip(distances,y))
        d_y.sort()
        y_sorted = [item[1] for item in d_y]
        return y_sorted[:self.k]
    
    def __call__(self, data, new_points):
        X, y = data
        retval = []
        for new_point in new_points :
            neighbours_labels = self._get_k_nearest_neighbours(X,y,new_point)
            retval.append(utils.majority_vote(neighbours_labels))
        return retval