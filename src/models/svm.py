from models.modelAdvanced import ModelAdvanced

import numpy as np
from sklearn.svm import SVC

class SvmModel(ModelAdvanced):

    def __init__(self, generatedData, features):
        super().__init__(generatedData, features)
        self.model = SVC()
        self.param_grid = {
            "kernel" : ['rbf', 'sigmoid'],
            "C" : np.geomspace(0.01, 1, num=2),
            "gamma": np.geomspace(0.001, 0.01, num=4),
            "coef0": np.linspace(20, 15, num=2), # This parameters contributes in kernel='sigmoid'
            "probability": [True]
        }


