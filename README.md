# Binary Machine Learning Classifiers
A collection of binary machine learning classifiers includes (KNN)

**KNN** (Fast_KNN.py)

A classifier that converts passed .csv file path or list to a 2D numpy array of size N * 3. Then classifies one or many anonymous data points specified by the user, using the passed *K* parameter which is an integer less than the size of the training data (Recommended to be between 10% to 20% of the size of the Training data to achieve the highest accuracy).
Fast_KNN.py offers two methods:

_classify(training_data, test_example, k)_

_classify_many(training_data, testing_data, k, get_confidence=False)_

*Real example screenshot on a data set with size 1000 and _K_=15

![image](https://user-images.githubusercontent.com/31454258/54305706-e2183300-45d8-11e9-8953-c5d98dd32160.png)
**____________________________________________________________________________________________________________________________________**

**Perceptron** (Perceptron.py)

A classifier that lineary seperates passed .csv file path or list to a 2D numpy array of size N * 3. Then sets the weights of the hypothetical line to be drawn which can be accessed from inside the script itself.
Perceptron.py offers one method *for now*:

_train(training_data, epochs=3, learning_rate=0.8, decaying_learning_rate=False)_

*Real example screenshot on a data set with size 500 and learning_rate=1, decaying_learning_rate=True

![image](https://user-images.githubusercontent.com/31454258/55142433-a8553980-514d-11e9-98d7-ef8f9a7322b2.png)
