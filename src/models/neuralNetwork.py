import numpy as np

from sklearn.neural_network import MLPClassifier
from models.modelAdvanced import ModelAdvanced


class NeuralnetworkModel(ModelAdvanced):

    def __init__(self, generatedData, features):
        super().__init__(generatedData, features)
        self.model = MLPClassifier(max_iter=500)
        self.param_grid = {            
            'activation': ['identity'],  # You can choose one of these: 'logistic', 'tanh', 'relu'
            'solver': ['adam'],  # You can choose either 'lbfgs','sgd' or by default 'adam'
            'learning_rate_init': np.linspace(0.1, 1, num=2) # It is used only with solver='adam' or 'sgd'
        }

