from random import randint
import numpy as np
import os


def __check(data):
    t = type(data)
    if t == str:
        if data[-4:] != ".csv":
            print("Only .csv is allowed for a data set path!")
            return None
        if os._exists(data):
            data = np.genfromtxt(data, delimiter=",")
            return data
        else:
            print(data, "doesn't exist!")
            return None
    elif t == np.ndarray or t == list:
        if t == list:
            data = np.array(data)
        if len(data) == 0:
            print("Data is empty!")
            return None
        elif len(data[0]) != 3:
            print("Array should have 3 columns (Feature, Feature, Label)!")
            return None
    else:
        print("Only .csv files, numpy.array, list are allowed as data sets")
        return None

    for d in data:
        for i in range(len(d)):
            i = 2-i
            if i == 2 and d[i] != 1 and d[i] != 0:
                print("Only type boolean or 1's and 0's is allowed in labels!")
                return None
            elif i < 2 and type(d[i]) != np.int32 and type(d[i]) != int:
                print("Only type 'int/float' is allowed in features!")
                print("Error in {", d, "}")
                return None

    return data


def classify_many(training_data, testing_data, k, get_confidence=False):
    """
    :param training_data: The data to test on. 2D Numpy Array with size N * 3
    :param testing_data: The data to classify. 2D Numpy Array with size N * 2 (Recommended size N * 3)
    :param k: Number of neighbours to consider in classifying test examples. int
    :param get_confidence: flag set True if confidence is needed in return *Optional
    :return: [Accuracy, *Confidence]. [int, *int]
    """

    if __check(training_data) is None:
        return None
    elif __check(testing_data) is None:
        return None
    elif type(k) != int:
        print("K parameter should be of type 'int'")
        return None
    elif k > len(training_data):
        print("K should be less than the size of the training data")
        return None
    hits = 0
    misses = 0
    total_confidence = 0
    for test in testing_data:
        output = classify(training_data, test, k)
        if get_confidence:
            [expected, confidence] = output
        else:
            expected = output
        if get_confidence:
            total_confidence = total_confidence + confidence
        if expected == test[2]:
            hits = hits + 1
        else:
            misses = misses + 1
    return [hits * 100 / (hits + misses), total_confidence * 100 / len(testing_data)] if get_confidence else hits * 100 / (
                hits + misses)


def classify(training_data, test_example, k):
    """
    :param training_data: The data to test on. 2D Numpy Array with size N * 3
    :param test_example: The example to classify. 1D Numpy Array with size 1 * 2 (Recommended size 1 * 3)
    :param k: Number of neighbours to consider in classifying test examples. int
    :return: [Expected classification, *Confidence]. [int, *int]
    """
    if __check(training_data) is None:
        return None
    elif __check(test_example) is None:
        return None
    elif type(k) != int:
        print("K parameter should be of type 'int'")
        return None
    elif k > len(training_data):
        print("K should be less than the size of the training data")
        return None

    points_distance = np.array([[0, 0]])  # 2D Numpy array [point index, euc distance]
    points_distance = points_distance[1:]
    nears = np.array([[0, 0]])  # 2D Numpy array [point index, euc distance]
    nears = nears[1:]

    for i in range(len(training_data[:])):
        euclidean_dist = np.sqrt(sum((test_example[0:2] - training_data[i, 0:2]) ** 2))
        element = np.array([[i, euclidean_dist]])
        points_distance = np.concatenate([points_distance, element])

    for i in range(len(points_distance[:])):
        if i < k:
            element = np.array([points_distance[i]])
            nears = np.concatenate([nears, element])
        elif points_distance[i, 1] < max(nears[:, 1]):
            index_of_max = nears[:, 1].tolist().index(max(nears[:, 1]).tolist())
            nears[index_of_max] = points_distance[i]

    [expected, confidence] = __get_expected(training_data, nears)
    return [expected, confidence]


def __get_expected(data, nears):
    """
    :param data: Used to get the value of indices stored in "nears".  2D Numpy Array with size N * 3
    :param nears: indices and distance of the nearest points to a point.  2D Numpy Array with size N * 2
    :return: [Expected classification, *Confidence]
    """
    positives = 0
    negatives = 0

    for index in nears[:, 0]:
        if int(data[int(index)][2]) == 1:
            positives = positives + 1
        elif int(data[int(index)][2]) == 0:
            negatives = negatives + 1
    if positives > negatives:
        expected = 1
    elif positives == negatives:
        expected = randint(0, 1)
    else:
        expected = 0
    confidence = positives / (positives + negatives) if expected == 1 else negatives / (positives + negatives)
    return [expected, confidence]
