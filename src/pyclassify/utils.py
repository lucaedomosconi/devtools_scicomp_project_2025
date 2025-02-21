import os
import yaml
import numpy as np
from line_profiler import profile
from numba import njit, prange
from numba.pycc import CC


cc = CC('utils_compiled')
@njit(parallel=True)
@cc.export('distance_numba', 'float64(float64[:],float64[:])')
def distance_numba(point1, point2) :
    retval = 0.0
    for i in prange(len(point1)) :
        retval += (point1[i] - point2[i]) ** 2
    return retval

if __name__ == "__main__" :
    cc.compile()

@profile
def distance(point1, point2) :
    """
    Calculate the squared Euclidean distance between two points.

    Parameters:
    point1 (list or tuple of floats): The first point in n-dimensional space.
    point2 (list or tuple of floats): The second point in n-dimensional space.

    Returns:
    float: The squared Euclidean distance between point1 and point2.

    Example:
    >>> distance([1, 2], [4, 6])
    25.0
    """
    retval = 0.0
    for i in range(len(point1)) :
        retval += (point1[i] - point2[i]) * (point1[i] - point2[i])
    return retval

@profile
def distance_numpy(point1_np : np.ndarray, point2_np : np.ndarray) -> float :
    """
    Calculate the squared Euclidean distance between two points using NumPy.

    Parameters:
    point1_np (np.ndarray): The first point in n-dimensional space.
    point2_np (np.ndarray): The second point in n-dimensional space.

    Returns:
    float: The squared Euclidean distance between point1_np and point2_np.

    Example:
    >>> import numpy as np
    >>> point1 = np.array([1.0, 2.0])
    >>> point2 = np.array([4.0, 6.0])
    >>> distance_numpy(point1, point2)
    25.0
    """
    point2_1_np = point2_np - point1_np
    return np.dot(point2_1_np, point2_1_np)

@profile
def majority_vote(neighbours : list) -> int :
    """
    Determines the majority vote from a list of neighbours.

    Args:
        neighbours (list): A list of votes (elements) to count.

    Returns:
        The element with the highest vote count. If there is a tie, 
        the first element with the highest count encountered in the list is returned.
    """
    vote_count = {}
    for vote in neighbours :
        if vote in vote_count :
            vote_count[vote] += 1
        else :
            vote_count[vote] = 1
    majority = None
    majority_count = 0
    for vote, count in vote_count.items() :
        if count > majority_count :
            majority_count = count
            majority = vote
    return majority

def read_config(file : str) -> dict :
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        file (str): The base name of the YAML file (without the .yaml extension).

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

def read_file(file_name) :
    labels = []
    features = []
    with open(file_name, 'r') as file :
        lines = file.readlines()
        features_and_labels = [line.strip().split(',') for line in lines]
        labels = [line[-1] for line in features_and_labels]
        if labels[0].isdigit() :
            labels = [int(item) for item in labels]
            print("labels are integers")
        else :
            print("labels are not integers")
        n = len(features_and_labels[0])
        features = [line[0:n-1] for line in features_and_labels]
        features_as_floats = []
        for line_as_list_of_str in features :
            line_as_list_of_floats = [float(line_str_elem) for line_str_elem in line_as_list_of_str]
            features_as_floats.append(line_as_list_of_floats)
    return features_as_floats, labels