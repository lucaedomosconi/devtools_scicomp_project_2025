import pyclassify.utils as utils
import pyclassify.classifier as classifier
import pytest
def test_distance() :
    point1 = [1.0,2.0,2.0]
    point2 = [4.0,6.0,2.0]
    point3 = [-0.5,1.3,3.0]
    point4 = [1.0,5.0,-3.0,9.8,10.4]
    point5 = [0.4,-8.1,-4.3,1.9,-2.5]
    assert utils.distance(point1,point1) == 0.0

    assert abs(utils.distance(point1,point2)-25.0) <= 1e-10

    assert abs((utils.distance(point4, point5) - utils.distance(point5, point4))) <= 1e-10

    assert utils.distance(point4, point5) >= 0.0

    assert utils.distance(point1, point2)**0.5 + utils.distance(point2,point3)**0.5 >= utils.distance(point1,point3)**0.5

    assert utils.distance(point1, point3)**0.5 + utils.distance(point1,point2)**0.5 >= utils.distance(point2,point3)**0.5

    assert utils.distance(point1, point3)**0.5 + utils.distance(point2,point3)**0.5 >= utils.distance(point1,point2)**0.5

def test_majority_vote() :
    dataset1 = [0,1,0,0,1,1,0,0,1,0,1,0,0,0]
    dataset2 = [1,0,3,3,2,4,1,2,3,2,5,1,2,4,1,5,3,4,1,2,3,1,0]
    dataset3 = ['b','g','b','g','g','g','b','b','b']
    assert utils.majority_vote(dataset1) == 0
    assert utils.majority_vote(dataset2) == 1
    assert utils.majority_vote(dataset3) == 'b'

def test_constructor() :
    classifier_0 = classifier.kNN(4)
    with pytest.raises(TypeError) :
        classifier_1 = classifier.kNN("1")
    with pytest.raises(TypeError) :
        classifier_2 = classifier.kNN([1,2])
    with pytest.raises(TypeError) :
        classifier_3 = classifier.kNN(2.4)
