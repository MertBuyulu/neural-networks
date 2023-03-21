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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self, dataFile, header=None):
        self.raw_input = pd.read_csv(dataFile, header=header, names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])

    def preprocess(self):
        # dataframe
        df = self.raw_input

        # convert categorical values into numerical values
        df['sex'].replace(['M', 'F', 'I'], [0, 1, 3], inplace=True)
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
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/MertBuyulu/neural-networks/main/abalone.data") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
