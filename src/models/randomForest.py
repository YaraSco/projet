from sklearn.ensemble import RandomForestClassifier
from models.modelAdvanced import ModelAdvanced

import numpy as np

class RandomForestModel(ModelAdvanced):

    def __init__(self, generatedData, features):
        super().__init__(generatedData, features)
        self.model = RandomForestClassifier()
        self.param_grid = {
            "max_depth": np.linspace(10, 60, num = 10),
            "n_estimators":range(80, 90) 
        }