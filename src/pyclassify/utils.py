import os
import yaml

def distance(point1, point2) :
    retval = 0.0
    for i in range(len(point1)) :
        retval += (point1[i] - point2[i]) * (point1[i] - point2[i])
    return retval

def majority_vote(neighbours) :
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

def read_config(file):
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
        n = len(features_and_labels[0])
        features = [line[0:n-1] for line in features_and_labels]
        features_as_floats = []
        for line_as_list_of_str in features :
            line_as_list_of_floats = [float(line_str_elem) for line_str_elem in line_as_list_of_str]
            features_as_floats.append(line_as_list_of_floats)
    return features_as_floats, labels