#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#   in the README file.
#
#####################################################################################################################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


class NeuralNet:
    def __init__(self, dataFile, header=None):
        self.raw_input = pd.read_csv(dataFile, header=header, names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])

    def preprocess(self):
        # dataframe
        df = self.raw_input

        # convert categorical values into numerical values
        df['sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)
        X = df.drop(['rings'], axis = 1)
        Y = df['rings']

        # normalize all values
        X_cols = X.columns
        s = StandardScaler()
        X = pd.DataFrame(s.fit(X).fit_transform(X), columns=X_cols)

        # split data into train and test sets 80/20
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

        return X_train, X_test, Y_train, Y_test

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        X_train, X_test, y_train, y_test = self.preprocess()

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        hyperparams = list(itertools.product(activations, learning_rate, max_iterations, num_hidden_layers))
        results = pd.DataFrame(columns=['Hyperparameters', 'Training Accuracy', 'Training Error', 'Test Accuracy', 'Test Error'])
        fig, ax = plt.subplots(2, 2, figsize=(18,12))
        for i, params in enumerate(hyperparams):
            activation, learning_rate, max_iter, num_hidden_layers = params
            nn = MLPClassifier(hidden_layer_sizes=tuple([50] * num_hidden_layers), activation=activation, learning_rate_init=learning_rate, max_iter=max_iter, early_stopping=False)

            num_classes = np.unique(y_train)
            train_history = []
            test_history = []

            for epoch in range(0, max_iter):
                nn.partial_fit(X_train, y_train, classes=num_classes)
                train_pred = nn.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)
                test_pred = nn.predict(X_test)
                test_accuracy = accuracy_score(y_test, test_pred)

                train_history.append(train_accuracy)
                test_history.append(test_accuracy)

            training_pred = nn.predict(X_train)
            test_pred = nn.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_pred)
            training_error = mean_squared_error(y_train, training_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            test_error = mean_squared_error(y_test, test_pred)
            results.loc[i] = [params, training_accuracy, training_error, test_accuracy, test_error]

            # Plot model history
            if max_iter == 100:
                ax[0, 0].plot(train_history, label=f"{params}")
                ax[0, 1].plot(test_history, label=f"{params}")
            else:
                ax[1, 0].plot(train_history, label=f"{params}")
                ax[1, 1].plot(test_history, label=f"{params}")
        
        # Format and show plot
        ax[0, 0].set_xlabel('Epochs')
        ax[0, 0].set_ylabel('Train Accuracy')
        ax[0, 0].set_title('Model Train Performance vs Number of Epochs (100 epochs)')
        ax[0, 0].legend()
        ax[0, 1].set_xlabel('Epochs')
        ax[0, 1].set_ylabel('Test Accuracy')
        ax[0, 1].set_title('Model Test Performance vs Number of Epochs (100 epochs)')
        ax[0, 1].legend()
        ax[1, 0].set_xlabel('Epochs')
        ax[1, 0].set_ylabel('Train Accuracy')
        ax[1, 0].set_title('Model Train Performance vs Number of Epochs (200 epochs)')
        ax[1, 0].legend()
        ax[1, 1].set_xlabel('Epochs')
        ax[1, 1].set_ylabel('Test Accuracy')
        ax[1, 1].set_title('Model Test Performance vs Number of Epochs (200 epochs)')
        ax[1, 1].legend()
        plt.show()
        
        # Output results table
        print(results)

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0


if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/MertBuyulu/neural-networks/main/abalone.data") # put in path to your file
    neural_network.train_evaluate()
