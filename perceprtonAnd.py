import numpy as np
import random

features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

def score(weights, bias, features):
    return np.dot(features, weights) + bias
    
def step(x):
    if x >= 0:
        return 1
    else:
        return 0
    
def prediction(weights, bias, features):
    return step(score(weights, bias, features))

#если прогноз отличается от метки, то точка классифицируется неправильно, а значит
# ошибка равна абсолютному значению оценки
def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(score(weights, bias, features))

def mean_error(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])
    return total_error/len(features)
    
def perceptron_trick(weights, bias, features, label, rate = 0.1):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label-pred) * features[i] * rate
    bias += (label - pred) * rate
    return weights, bias
    
def perceptron_algorithm(features, labels, rate = 0.1, epochs = 100):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    
    for epoch in range(epochs):
        error = mean_error(weights, bias, features, labels)
        errors.append(error)
        i = random.randint(0, len(features)-1)
        weights, bias = perceptron_trick(weights, bias, features[i], labels[i])
    return weights, bias, errors

# получаем веса и смещение 
w, c, e = perceptron_algorithm(features, labels)
#print(f"Веса: {w}, Смещение: {c}, Ошибки: {e}")

# у меня не выходит сделать циклом фор я хз поч
print("Результаты работы персептрона:")
print(prediction(w, c, features[0]))
print(prediction(w, c, features[1]))
print(prediction(w, c, features[2]))
print(prediction(w, c, features[3]))
    