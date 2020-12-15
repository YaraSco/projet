from models.modelAdvanced import ModelAdvanced
from sklearn import ensemble
from sklearn import tree

import numpy as np

class AdaBoostModel(ModelAdvanced):

    def __init__(self, generatedData, features):
        super().__init__(generatedData, features)
        self.model = ensemble.AdaBoostClassifier()
        self.param_grid = {
            "base_estimator" : [tree.DecisionTreeClassifier(max_depth=n) for n in range(10, 15)],
            "n_estimators" : range(80, 85),
            "learning_rate" : np.geomspace(0.01, 1, num=2)
        }