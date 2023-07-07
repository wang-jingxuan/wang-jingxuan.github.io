# Support Vector Machine(SVM) Algorithm

## 1. Fire up


```python
# import sklearn SVC

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
from numpy import sin, cos, pi, tanh, exp
```

## 2. Training data generation

Why 2-dim data? For support vector visualization


```python
max r = 0.45
x1 = []
x2 = []
y = []
for i in range(20):
    r = max_r * np.random.rand()
    rand_num = np.random.rand ()
    x1.append(0.5 + r * cos(2 * pi * rand_num))
    x2.append(0.5 + r * sin(2 * pi * rand_num))
    y.append(1)

for i in range(20):
    r = max_r * np.random.rand()
    rand_num = np.random.rand()
    x1.append(-0.5 + r * cos(2 * pi * rand_num))
    x2.append(-0.5 + r * sin(2 * pi * rand_num))
    y.append (-1)

dict1 = {"xl": x1, "x2": x2}
training_X = pd.DataFrame(dict1)
dict2 = {"y": y}
training_y = pd.DataFrame(dict2)
training_X.head(5)
```


```python
training_y.tail(5)
```


```python
plt. figure(figsize = (10, 5))

plt. scatter(x1[0 : 20], x2[0 : 20], color = "black", label = "Class 1")
#plt.scatter(x1[5], x2[5], color = "blue", label = "Class 1 S V 1")
#plt.scatter(x1[10], x2[10], cotor =“btuel', label = "Class 1 S V 2")

plt.scatter(x1[20: 40], x2[20 : 40], color = "red", label = "Class -1")
#plt.scatter(x1[39], x2[39], color = "pink", label = "Class -1 S V 1")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()
```

## 3. Kernel fuction design


```python
def K(vecl, vecz, kernel_function = "rbf"， Gamma = 0.2, r = 0.2):#vecl and vec 2 are transformed into row vect
    vec1 = np.array(vec1).reshape(1,-1)
    vec2 = np.array(vec2).reshape(1,-1)
    
    if kernel_function == "rbf":
        return torch.FloatTensor([np.exp(- Gamma * np.linalg.norm(vec1 - vec2, ord = 2) ** 2)])
    
    elif kernel_fuction == "sigmoid":
        return torch.FloatTensor([tanh(Gamma * np.matmul(vec1, vec2.T) = r)[0][0]])
```

## 4. Training the SVM model by gradient descent method

(with the help of the Automatic Differentiation function in Pytorch)

(Including how to solve the trouble of gradient disruption in Pytorch)


```python
C = 10.0 # penalty factor
M = training_X.__len__()
alpha_lack = Variable(torch.FloatTensor(n.zeros((1, M - 1))), requires_grad = True) # alpha_new
y = torch.FloatTensor(training_y.iloc[:, 0]).reshape(-1, 1)
```


```python
print(alpha_lack.shape)
alpha_lack
```


```python
print(y.shape)
y.T
```


```python
Iter times = 10000000
lr = 0.22 # learning rate
Obj_List = []

for it in range(Iter times):
    
    alpha_last = - torch.mm(alpha_lack, y[0 : M - 1, 0].reshape(-1, 1)) / y[M - 1]
    
    sum_ = 0.0
    
    for i in range(M):
        for j in range (M):
            
            if i != M - 1 and j != M - 1:
                sum_ = sum_ + 0.5 * alpha_lack[0, i] * alpha_lack[0, j] * y[i, 0] * y[j, 0] * K(training_X.iloc[i, :], training_X.iloc[j, :])
            
            elif i == M - 1 and j !=M - 1:
                sum_ = sum_ + 0.5 * alpha_last * alpha_lack[0, j] * y[i, 0] * y[j, 0] * K(training_X.iloc[i, :],training_X.iloc[j, :])
            
            elif i != M - 1 and j == M - 1:
                sum_ = sum_ + 0.5 * alpha_last * alpha_lack[0, i] * y[i, 0] * y[j, 0] * K(training_X.iloc[i, :],training_X.iloc[j, :])

            else:
                sum_ = sum_ + 0.5 * alpha_last * alpha_last * y[i, 0] * y[j, 0] * K(training_X.iloc[i, :], training_X.iloc[j, :])

    Obj = sum_ - (torch.sum(alpha_lack) + alpha_last)
    
    Obj.backward()
    
    grad = alpha_lack.grad.data
    
    lr_temp = lr / (1 + it * 0.0008)
    
    alpha_lack.data = alpha_lack.data - lr_temp * grad
    
    alpha_lack.grad.data.zero_()
    
    for col in range(alpha_lack.shape[1]):  # clipping
        
        if alpha_lack[0, col] > C:
            alpha_lack[0, col] = C
        
        elif alpha_lack[0, col] < 0.0:
            alpha_lack[0, col] = 0.0
    
    alpha_lack = Variable(torch.FloatTensor(alpha_lack), requires_grad = True)
    
    Obj_List.append(Obj)
    
    print("-----------------------------------------------------------------------")
    print(it + 1, "iterations has been completed!")
    print("    -> Current Obj = ", Obj)
    print("    -> Current alpha_lack = ", alpha_lack)
    print("-----------------------------------------------------------------------")
```


```python
alpha_lack
```


```python
alpha_last = - torch.mm(alpha_lack, y[0 : M - 1, 0].reshape(-1, 1)) / y [M - 1]
alpha_last
```

## 5. Support vectors visualization


```python

```

## 6. Computing the solutions of the original problem: W & b


```python
s = 5
ys = y[s, 0]
sum_record = 0.0

for i in range(M - 1):
    sum_record = sum_record + alpha_lack[0, i] * y[i, 0] * K(training_X.iloc[i, :], training_X.iloc[s, :1])

sum_record = sum_record + alpha_last * y [M - 1, 0] * K(training_X.iloc[M - 1, :], training_X.iloc[s, :])

b = (1 / ys) - sum_record
b
```


```python
W = torch.FloatTensor(np.zeros ((2, 1)))
W
```


```python
for i in range(M - 1):
    W = W + alpha_lack[0, i] * y[i, 0] * torch.FloatTensor(n.array(training_X.iloc[i, :]).reshape (-1, 1))

W = W + alpha_last * y [M - 1, 0] * torch.FloatTensor(np.array(training_X.array.iloc[M - 1, :]).reshape(-1, 1))
W
```

## 7. Prediction and model evaluation


```python
def prediction(x):
    sum_record = 0.0
    for i in range(M - 1):
        sum_record = sum_record + alpha_lack[0, i] * y[i, 0] * K(training_X.iloc[i, :], x)
        sum_record = sum_record + alpha_last * y [M - 1, 0] * K(training_X.iloc[M - 1, :], x)
        
    return sum record + b
```


```python
max_r = 0.45
x1 = []
x2 = []
y test = [] 

for i in range(50):
    r = max_r * np.random.rand()
    rand_num = np.random.rand ()
    x1.append(%.5 + r * cos(2 * pi * rand_num))
    x2.append(0.5 + r * sin(2 * pi * rand_num))
    y_test.append(1)

for i in range(50):
    r = max_r * np.random.rand()
    rand_num = np random.rand()
    x1.append(-0.5 + r * cos (2 * pi * rand_num)) 
    x2.append(-0.5 + r * sin(2 * pi * rand num))
    y_test.append(-1)
    
dict1 = {"xl": x1, "X2": x2}
test_X = pd.DataFrame(dict1)

dict2 = {"y": y_test}
test_y = pd.DataFrame(dict2)

test_X.head (5)
```


```python
prediction_results = []

for i in range(test_X.__len__()):
    y_pred_i = prediction(test_X.iloc[i, :])
    prediction_results.append(y_pred_i)

for i in range(prediction_results.__len__()):
    
    if prediction_results[i] > 0.0:
        prediction_results[i] = 1.0
    
    else:
        prediction_results[i] = -1.0

Matrix(np.array(prediction_results).reshape(-1, 10))
```


```python
from sklearn.metrics import accuracy_score
print("The accuracy score in this prediction is", 
      accuracy_score(np.array (prediction_results), test_y) * 100, "%")
```


```python

```
