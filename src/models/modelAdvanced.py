# -*- coding: utf-8 -*-

from models.model import Model
from sklearn.model_selection import GridSearchCV


class ModelAdvanced(Model):
    """
    This class inherits from model and uses exhaustive search 
    to optimize the hyperparameters of each model
    """

    def __init__(self, generatedData, features, k=4):
        """
        Init any Classifier with hyper parameters
        :param k: number of folds for cross-validation
        """
        super().__init__(generatedData, None, features)
        self.k = k

    def train(self, verbose=False):
        """
        This method is overridden of train to optimize the hyperparameters using grid search and cross-validation
        :param verbose: whether to show verbosity for cross validation or not
        """
        # refit=True gives the best_estimator_ and the metric evaluation like best_params_ and best_score_
        grid_search = GridSearchCV(self.model, self.param_grid,
                                    n_jobs=-1, cv=self.k, iid=False, refit=True, verbose=verbose)

        self.classif = grid_search.fit(self.X_train, self.t_train).best_estimator_

        if verbose:
            print("The optimized hyperparameters are {} and their scores are {:.2f}"
                  .format(grid_search.best_params_, grid_search.best_score_))

        # Do the training with the optimized hyperparameters
        super().train()
