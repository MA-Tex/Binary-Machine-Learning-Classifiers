import random
import numpy as np
import os
import matplotlib.pyplot as plt

def __check(data):
    t = type(data)
    if t == str:
        if data[-4:] != ".csv":
            print("Only .csv is allowed for a data set path!")
            return None
        if os.path.exists(data):
            data = np.genfromtxt(data, delimiter=",")
            np.random.shuffle(data)
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
            elif i < 2 and type(d[i]) != np.int32 and type(d[i]) != int and type(d[i]) != np.float64:
                print("Only type 'int/float' is allowed in features!")
                print("Error in {", d, "}")
                print(type(d))
                return None
    return data


weights = np.array([0])
weights = weights[:-1]


def train(training_data, epochs=3, learning_rate=0.8, decaying_learning_rate=False):
    training_data = __check(training_data)
    if training_data is None:
        return None

    global weights

    for i in range(len(training_data[0])):
        rand_weight = random.random()
        weights = np.append(weights, rand_weight)

    if epochs < 1:
        epochs = 1

    total = epochs*len(training_data)
    for epoch in range(round(epochs)):
        for i in range(len(training_data)):
            if decaying_learning_rate:
                current_iteration = epoch*len(training_data) + i
                learning_rate = learning_rate * (total-current_iteration)/total
            expected_label = __get_expected(weights, training_data[i])
            if training_data[i][-1] != expected_label:
                constant = learning_rate*(training_data[i][-1] - expected_label)
                weights = weights + (constant * np.append(1, training_data[i][:-1]))
    return training_data


def __get_expected(w, test_data):
    num = sum(w * np.append(1, test_data[:-1]))
    return np.sign(num) if num >= 0 else 0


def __plot_seperator(dots):
    if dots is None:
        return None
    plt.scatter(dots[:, 0], dots[:, 1], c=dots[:, 2])
    lower_range = min(min(dots[:, 0]), min(dots[:, 1]))
    max_range = max(max(dots[:, 0]), max(dots[:, 1]))
    x1 = np.linspace(lower_range, max_range, 100)
    x2 = -(x1*weights[1] + 1*weights[0])/weights[2]
    plt.plot(x1, x2, label='separator')
    plt.show()


data = "test_data_set_perceptron.csv"
data = train(data, learning_rate=1, decaying_learning_rate=True)
__plot_seperator(data)
