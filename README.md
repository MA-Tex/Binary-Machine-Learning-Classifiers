# Binary Machine Learning Classifiers
A collection of binary machine learning classifiers includes (KNN)

**KNN** (Fast_KNN.py)

A classifier that converts passed .csv file path or list to a 2D numpy array of size N * 3. Then classifies one or many anonymous data points specified by the user, using the passed *K* parameter which is an integer less than the size of the training data (Recommended to be between 10% to 20% of the size of the Training data to achieve the highest accuracy).
Fast_KNN.py offers two methods:

_classify(training_data, test_example, k)_

_classify_many(training_data, testing_data, k, get_confidence=False)_

*Real example screenshot on a data set with size 1000 and _K_=15

![image](https://user-images.githubusercontent.com/31454258/54305706-e2183300-45d8-11e9-8953-c5d98dd32160.png)
