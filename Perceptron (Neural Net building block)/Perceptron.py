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


weights = np.array([0])
weights = weights[:-1]


def train(training_data, epochs=3, learning_rate=0.5):
    if __check(training_data) is None:
        return None

    global weights

    for i in range(len(training_data[0][:-1])+1):
        rand_weight = randint(0, 1)
        weights = np.append(weights, rand_weight)

    print(len(weights))
    if epochs < 1:
        epochs = 1
    for i in range(round(epochs)):
        for d in training_data:
            expected_label = __get_expected(weights, d)
            if d[-1] != expected_label:
                constant = learning_rate*(d[-1] - expected_label)
                weights = weights + (constant * np.append(1, training_data[:-1]))


def __get_expected(w, test_data):
    num = sum(w * np.append(1, test_data[:-1]))
    return np.sign(num) if num != 0 else test_data[-1]


a = [[0,  1,    1],
     [1,  0,    1],
     [0,  -1,   0],
     [-1, 0,    0]]

#train(a)
#print(weights)
