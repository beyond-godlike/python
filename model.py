import numpy as np
import matplotlib.pyplot as plt


s = np.array([100, 200, 200, 250, 325])
p = np.array([200, 475, 400, 520, 735])

def predict(s):
    return 2*s + 50

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

#for element in s:
#    print(predict(element))
    
p2 = np.array([predict(element) for element in s])


mae = mean_absolute_error(p, p2)
print("Средняя абсолютная погрешность:", mae)

rmse = mean_squared_error(p, p2)
print("Средняя квадратичная погрешность:", rmse)
