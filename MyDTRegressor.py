import numpy as np

class MyDTRegressor:
    def __init__(self, max_depth = 5, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    class Node:
        def __init__(self, feature_index = None, treshold = None, left = None, right = None, value = None):
            self.feature_index = feature_index
            self.treshold = treshold
            self.left = left
            self.right = right
            self.value = value
            
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth = 0)
            
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
            
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return self.Node(value = np.mean(y))
            
        best_feature, best_treshold = None, None
        best_mse = float('inf') 
        best_splits = None
        
        for feature_index in range(num_features):
            tresholds = np.unique(X[:, feature_index])
            
            for treshold in tresholds:
                left_indices = X[:, feature_index] <= treshold
                right_indices = X[:, feature_index] > treshold
                
                if(len(y[left_indices]) == 0 or len (y[right_indices]) == 0):
                    continue
                
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_treshold = treshold
                    best_splits = (left_indices, right_indices)
                    
            if best_feature is None:
                return self.Node(value = np.mean(y))
                
            left = self._build_tree(X[best_splits[0]], y[best_splits[0]], depth + 1)
            right = self._build_tree(X[best_splits[1]], y[best_splits[1]], depth + 1)
            
            return self.Node(feature_index = best_feature, treshold = best_treshold, left = left, right = right)
            
    def _calculate_mse(self, y_left, y_right):
        def mse(y):
            if len(y) == 0:
                return 0
            return np.mean((y - np.mean(y)) ** 2)
        return (len(y_left) * mse(y_left) + len(y_right) * mse(y_right)) / (len(y_left) + len(y_right))
        
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])
        
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.treshold:
            return self._predict_sample(x, node.left)
        else: 
            return self._predict_sample(x, node.right)
        
        
