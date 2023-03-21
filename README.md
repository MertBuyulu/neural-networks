# 4375 Assignment 2: Neural-Networks

This assignment uses the MLPClassifier from the sklearn library on an abalone data set to predict the number of rings in the abalone's shell. Generally, the only way to find out the age of an abalone is to manually count the number of rings in its shell, a tedious and time consuming task. In this project, we try 24 combinations of hyperparameters and compare their training and testing results presented in plots and a table. The plots show the accuracy history of the model through each epoch for both training and testing. The features used from this dataset are: sex, length, diameter, height, whole weight, shucked weight, viscera weight, and shell weight.

## Execution Instructions

This program runs with `Python 3.9.12` and above.

The following imports are required to run `NeuralNet.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
```

Run the program using any acceptable terminal or shell with:

```bash
python NeuralNet.py
```

## Dataset

The abalone dataset is hosted on our GitHub repository at:

```text
https://raw.githubusercontent.com/MertBuyulu/neural-networks/main/abalone.data
```

The dataset contains `4177` instances with `8` attributes per instance.

## Output

Running `NeuralNet.py` will output a plot window with the four plots containing the model history for each combination of hyperparameters. When the window is closed, the program will print a table of training and test accuracies and errors for each combination of hyperparameters.

An example of the table outputted after running `NeuralNet.py`:

```text
             Hyperparameters  Training Accuracy  Training Error  Test Accuracy  Test Error
0   (logistic, 0.01, 100, 2)           0.294223        4.895840       0.265550    5.708134
1   (logistic, 0.01, 100, 3)           0.297216        4.910207       0.244019    5.654306
2   (logistic, 0.01, 200, 2)           0.315774        4.630650       0.244019    5.863636
3   (logistic, 0.01, 200, 3)           0.320563        5.051482       0.266746    5.558612
4    (logistic, 0.1, 100, 2)           0.315774        4.827597       0.255981    5.964115
5    (logistic, 0.1, 100, 3)           0.289434        5.013768       0.257177    5.790670
6    (logistic, 0.1, 200, 2)           0.328644        4.072733       0.222488    6.251196
7    (logistic, 0.1, 200, 3)           0.276265        4.296917       0.218900    6.377990
8       (tanh, 0.01, 100, 2)           0.391799        4.384017       0.247608    6.867225
9       (tanh, 0.01, 100, 3)           0.532176        3.181981       0.226077    7.370813
10      (tanh, 0.01, 200, 2)           0.471116        3.499252       0.228469    8.179426
11      (tanh, 0.01, 200, 3)           0.656989        2.523496       0.220096    6.887560
12       (tanh, 0.1, 100, 2)           0.236157        6.088896       0.191388    6.826555
13       (tanh, 0.1, 100, 3)           0.237653        7.983837       0.217703    8.465311
14       (tanh, 0.1, 200, 2)           0.281054        6.846154       0.220096    8.273923
15       (tanh, 0.1, 200, 3)           0.198444        9.055672       0.173445    9.338517
16      (relu, 0.01, 100, 2)           0.359174        4.317570       0.242823    6.342105
17      (relu, 0.01, 100, 3)           0.445076        3.566896       0.227273    8.327751
18      (relu, 0.01, 200, 2)           0.458545        3.395989       0.245215    8.491627
19      (relu, 0.01, 200, 3)           0.519006        2.686022       0.200957    8.063397
20       (relu, 0.1, 100, 2)           0.263095        4.936247       0.234450    5.872010
21       (relu, 0.1, 100, 3)           0.296917        4.799461       0.248804    5.265550
22       (relu, 0.1, 200, 2)           0.287339        5.171506       0.259569    5.990431
23       (relu, 0.1, 200, 3)           0.275965        5.252320       0.250000    5.767943
```

See the report for details discussing the best hyperparameter selection, outputs, and plots.