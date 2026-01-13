import numpy as np
from sklearn.model_selection import KFold

class DataSplitter:
    def __init__(self, x, y, num_folds=5, random_seed=42):
        self.x = np.array(x)
        self.y = np.array(y)
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_seed)

    def get_splits(self):
        """
        Returns train-test splits for K-Fold cross-validation.
        """
        for train_index, test_index in self.kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            yield x_train, x_test, y_train, y_test, train_index, test_index
