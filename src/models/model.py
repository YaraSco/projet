from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

class Model:
    def __init__(self, GeneratedData, model, features=False):
        self.gd = GeneratedData
        self.classif = model

        if features:
            self.X_train = GeneratedData.features_data_train()
            self.X_test = GeneratedData.features_data_test()
        else:
            self.X_train = GeneratedData.generated_data_train()
            self.X_test = GeneratedData.generated_data_test()

        self.t_train = GeneratedData.generated_target_train()
        self.t_test = GeneratedData.generated_target_test()

    def train(self):
        """
        This method trains the classifier
        """
        self.classif.fit(self.X_train, self.t_train)

    def prediction(self, X, predict_proba=False):
        """
        make predictions using new data instance
        :param X: data given
        :return: prediction of data given
        """
        if predict_proba:
            prediction = self.classif.predict_proba(X)
        else:
            prediction = self.classif.predict(X)

        return prediction

    def _error(self ,X ,t):
        """
        error quantifiying the quality of prediction
        :param X: the input
        :param t: the target
        :return: score
        """
        error = self.classif.score(X ,t)
        return error

    def loss(self, X, t):
        """
        This method computes the cross entropy loss
        :param X: the input to be predicted
        :param t: the target
        :return: the model's cross entropy loss
        """
        return log_loss(t, self.prediction(X, predict_proba=True))

    def accuracy(self, X, t):
        """
        This methods computes the accuracy
        :param X: the input to be predicted
        :param t: the target
        :return: the model's accuracy
        """
        return accuracy_score(t, self.prediction(X))

    def display_evaluation(self):
        """
        this method displays the results' metrics (accuracy and loss) of a model
        """
        print("--- {} ---".format(self.classif.__class__.__name__))

        # Display the results of training the data
        print('Training : accuracy {:.4%}\tloss {:.4f}'.format(
            self.accuracy(self.X_train, self.t_train), self.loss(self.X_train, self.t_train) ))

        # Display the results of testing the data
        print('Test     : accuracy {:.4%}\tloss {:.4f}'.format(
            self.accuracy(self.X_test, self.t_test), self.loss(self.X_test, self.t_test) ))

        print("-------------------------------------------------------------------")
