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
python3 NeuralNet.py
```

## Dataset

The abalone dataset is hosted on our GitHub repository at:

```text
https://raw.githubusercontent.com/MertBuyulu/neural-networks/main/abalone.data
```

The dataset contains `4177` instances with `8` attributes per instance.

## Modifications to the Template Code

The only modifications to the template code that we made is in the main function, where we do not call the preprocess function explicitly because we call it inside of our train_evaluate function. No modifications were made to the hyperparameter grid. We only added code inside the preprocess and train_evaluate function.

## Output

Running `NeuralNet.py` will output a plot window with the four plots containing the model history for each combination of hyperparameters. When the window is closed, the program will print a table of training and test accuracies and errors for each combination of hyperparameters.

An example of the table outputted after running `NeuralNet.py`:

```text
             Hyperparameters  Training Accuracy  Training Error  Test Accuracy  Test Error
0   (logistic, 0.01, 100, 2)           0.305298        5.030530       0.253589    5.555024
1   (logistic, 0.01, 100, 3)           0.294822        5.284645       0.251196    6.043062
2   (logistic, 0.01, 200, 2)           0.324154        5.005986       0.239234    6.356459
3   (logistic, 0.01, 200, 3)           0.318468        4.962586       0.255981    6.033493
4    (logistic, 0.1, 100, 2)           0.332236        4.386411       0.250000    6.034689
5    (logistic, 0.1, 100, 3)           0.276564        4.947620       0.224880    5.930622
6    (logistic, 0.1, 200, 2)           0.337324        4.506435       0.227273    6.541866
7    (logistic, 0.1, 200, 3)           0.266986        6.582760       0.232057    7.485646
8       (tanh, 0.01, 100, 2)           0.392098        4.373242       0.221292    7.130383
9       (tanh, 0.01, 100, 3)           0.473212        3.663873       0.208134    7.482057
10      (tanh, 0.01, 200, 2)           0.484286        3.702484       0.227273    7.764354
11      (tanh, 0.01, 200, 3)           0.674648        1.873092       0.202153    8.453349
12       (tanh, 0.1, 100, 2)           0.257408        6.258007       0.252392    6.727273
13       (tanh, 0.1, 100, 3)           0.256211        6.008979       0.223684    6.736842
14       (tanh, 0.1, 200, 2)           0.288836        5.912900       0.205742    7.661483
15       (tanh, 0.1, 200, 3)           0.247531        7.279856       0.245215    7.789474
16      (relu, 0.01, 100, 2)           0.376235        4.313080       0.258373    7.082536
17      (relu, 0.01, 100, 3)           0.424424        4.042802       0.236842    7.514354
18      (relu, 0.01, 200, 2)           0.419934        3.950314       0.226077    7.565789
19      (relu, 0.01, 200, 3)           0.572284        2.791978       0.255981    8.793062
20       (relu, 0.1, 100, 2)           0.298414        5.311284       0.260766    5.911483
21       (relu, 0.1, 100, 3)           0.289434        4.541155       0.253589    5.467703
22       (relu, 0.1, 200, 2)           0.286142        5.081712       0.253589    6.145933
23       (relu, 0.1, 200, 3)           0.268482        5.258605       0.255981    5.710526
```

See the report for details discussing the best hyperparameter selection, outputs, and plots.