import numpy as np

from models.modelAdvanced import ModelAdvanced
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class GeneralDiscriminantAnalyseModel(ModelAdvanced):

    def __init__(self, generatedData, features):
        super().__init__(generatedData, features)
        self.model = LinearDiscriminantAnalysis()
        self.param_grid = {
            "solver": ['lsqr', 'eigen'], # lsqr means least squares solution
            "shrinkage": ['auto']
        }