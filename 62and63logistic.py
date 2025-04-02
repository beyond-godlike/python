import numpy as np
import random

features = np.array([[1,1]])
labels = np.array([0])

def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))

def score(weights, bias, features):
    return np.dot(weights, features) + bias

def prediction(weights, bias, features):
    return sigmoid(score(weights, bias, features))

def log_loss(weights, bias, features, label):
    pred = 1.0*prediction(weights, bias, features)
    return -label*np.log(pred) - (1-label)*np.log(1-pred)

def total_log_loss(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += log_loss(weights, bias, features[i], labels[i])
    return total_error

def logistic_trick(weights, bias, features, label, learning_rate = 0.1):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label-pred) * features[i] * learning_rate
    bias += (label-pred) * learning_rate
    return weights, bias

def logistic_regression_algorithm(features, labels, learning_rete = 0.1, epochs = 100):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features)-1)
        weights, bias = logistic_trick(weights, bias, features[j], labels[j])
    return weights, bias 

def classifier_before(p):
    # веса 2 и 3 смещение -4
    y = 2*p[0] + 3*p[1] - 4
    return sigmoid(y)

def classifier_after(weights, bias, p):
    y = weights[0]*p[0] + weights[1]*p[1] - bias
    return sigmoid(y)

def find_x_for_sigmoid(y):
    if not 0 < y < 1:
        return None
    return np.log(y / (1 - y))

def find_features(weights, bias, target_score):
    target = target_score - bias
    w1 = weights[0]
    w2 = weights[1]
    
    if w1 == 0 and w2 == 0:
        if target == 0:
            return np.array([0, 0])
        else:
            return None
    elif w2 != 0:
        f2 = target / w2
        return np.array([0, f2])  # Предполагаем f1 = 0
    elif w1 != 0:
        f1 = target / w1
        return np.array([f1, 0])  # Предполагаем f2 = 0
    else:
        return None

    #return np.dot(weights, features) + bias
    #w[1]*f[1]  + w[2]*f[2] + c = target_score
    #weights[1]*f[1] + weights[2]*f[2] = target_score - c
    #w2 = ( target_score - c) / weights[1]
    #return np.dot(0, w2)

p = np.array([1, 1])
#print(classifier_before(p))
#print(logistic_regression_algorithm(features, labels))
w, c = logistic_regression_algorithm(features, labels)
print(classifier_after(w, c, p))

target_y = 0.8
sc = find_x_for_sigmoid(target_y)

found_features = find_features(w, c, sc)
print(found_features)
print(prediction(w, c, found_features))
