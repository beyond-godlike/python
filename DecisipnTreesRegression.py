from sklearn.tree import DecisionTreeRegressor
import utils9
from MyDTRegressor import MyDTRegressor
import numpy as np


features = [[10], [20], [30], [40], [50], [60], [70], [80]]
labels = [7, 5, 7, 1, 2, 1, 5, 4]

dt_regressor = MyDTRegressor(max_depth = 2)
dt_regressor.fit(features, labels)

    

#dt_regressor1 = DecisionTreeRegressor(max_depth = 2)
#dt_regressor1.fit(features, labels)
#dt_regressor.score(features, labels)

utils9.plot_regressor(dt_regressor, features, labels)
#utils9.plot_regressor(dt_regressor1, features, labels)
#utils.display_tree(dt_regressor)
