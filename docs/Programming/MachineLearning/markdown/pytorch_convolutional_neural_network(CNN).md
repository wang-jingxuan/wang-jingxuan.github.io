
# Mist Image Classification by CNN

## 1. Fire Up


```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
import sympy
from sympy import Matrix
```

## 2. Data Generating


```python
mist dataset = keras.datasets.mnist
(data_train, label_train), (data_test, label_test) = mnist_dataset.load_data ()
```
