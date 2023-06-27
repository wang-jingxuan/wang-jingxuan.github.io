
# Mnist Image Classification by DNN

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


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[2], line 11
          9 from torchvision import datasets
         10 from torchvision import transforms
    ---> 11 import tensorflow as tf
         12 from tensorflow import keras
         13 import sympy


    ModuleNotFoundError: No module named 'tensorflow'


## 2. Loss Functions


```python
out = torch.FloatTensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]).reshape(2, 3)
print(out)
label = torch.LongTensor([2, 1])
print(label)
```


```python
loss_function = n.CrossEntropyLoss()
print(loss_function(out, label))
```


```python
a = p.log (np.exp(0.1) + np.exp(0.7) + np.exp(0.2)) - 0.7
print(a)
b = np.log(np.exp(0.2) + np.exp(0.3) + np.exp(0.5)) - 0.3
print (b)
print("loss =", (a + b) / 2)
```

## 3. Data Generating


```python
mnist_dataset = keras.datasets.mnist
(data_train, label_train), (data_test, label_test) = mist_dataset.load_data()
```


```python
# Number of Training Samples
print(data_train.__len__())
print (label_train.__len__())
# Number of Test Samples
print(data_test.__len__())
print(label_test.__len__())
```


```python
Matrix(data_train[0])
```


```python
plt.figure()

plt.subplot(121)
plt.imshow(data_train[0])
print("The label for image[0] is", label_train[0])

plt.subplot(122)
plt.imshow(data_train[59999])
print ("The label for image[59999] is", label_train [59999]) 

plt.show()
```


```python
# Convert numpy.ndarry data to tensor type in Pytorch
data_train = Variable(torch.FloatTensor(data_train)) / 255.0
data_test = Variable(torch.FloatTensor(data_test)) / 255.0
label_train = Variable(torch.LongTensor(label_train))
label_test = Variable(torch.LongTensor(label_test))
```


```python
# Convert a 28 × 28 Matrix into a 784 × 1 Vector 
data_train[0].reshape(1, -1).size()
```

## 4. Model Training


```python
class DeepNeuralNetworkModel(nn.Module):
    # Constructor of the class
    def __init__(self, input_dim1, output_dim1, input_dim2, output_dim2,
               input_dim3, output_dim3):
        super(DeepNeuralNetworkModel, self).__init__()
        
        # Fully Connected Layer 1
        self.FC_layer1 = nn.Linear(input_dim1, output_dim1)
        #nn.init.constant_(self.FC_layer1.weight, 0.1)
        #nn.init.constant_(self.FC_layer1.bias, -0.1)
        
        # Fully Connected Layer 2
        self.FC_layer2 = nn.Linear(input_dim2, output_dim2)
        
        # Fully Connected Layer 3
        self.FC_layer3 = nn.Linear(input_dim3, output_dim3)
        
        # Activation Function Sigmoid()
        self.act_sig = nn.Sigmoid()
        
        # Activation Function ReLU()
        self.act_relu = nn.ReLU()
        
    # Forward propagation function
    def forward(self, x):    # dim of x: N × input_diml
        
        z1_ = self.FC_layer1(x)
        z1 = self.act_sig(z1_)

        z2_ = self.FC_layer2(z1)
        z2 = self.act_sig(z2_)
        
        z3_ = self.FC_layer3(z2)
        # z3 = self.act_relu(z3_)
        
        return z3
```


```python
alpha = 0.5
DNN_Model = DeepNeuraletworkModel(784, 128, 128, 64, 64, 10)
optimizer = torch.optim.SGD(DNN_Model.parameters(), lr = alpha)
loss_function = nn.CrossEntropyLoss()

# Dynamically Change the learning rate
def adjust_learning_rate(optimizer, epoch):
    if epoch <= 100:
        lr = alpha
        
    elif epoch > 100:
        lr = alpha / (1 + 0.01 * (epoch - 100))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```


```python
epochs = 10000
batch size = 10000
batches = int((data_train.__Len__ ()) / batch size)
col = 28 * 28
loss_list = []
```


```python
for epoch in range(epochs):
    for batch in range(batches):
        
        # Get the input data matrix: dim = 100 × 784
        input_data = Variable(torch. FloatTensor(batch_size, col))
        for j in range(batch_size):
            input_data[j] = data_train[j + batch_size * batch].reshape(1, -1)
            
            # Forward propagation: output_data has dim = 100 × 10
            output_data = DNN_Model.forward(input_data)
            
            # Compute cross entropy loss
            loss = loss_function(output_data, label_train[batch_size * batch : batch_size * (batch + 1)])
            
            # backward propagation
            loss.backward()

            # update parameters
            optimizer.step()

            # Reset grad to 0
            optimizer.zero_grad()
            
            # Save loss for this batch
            loss_list.append(loss)

            # Print details for the gradient descent
            if(epoch) % 10 == 0 and (batch + 1) % 1 == 0:
                print("epoch =", epoch + 1, "; ", "batch =", batch + 1, ":")
                print("      -> Now loss =", loss.item())
                print("-------------------------------------------------------------")
            
        adjust_learning_rate(optimizer, epoch) 
        
        if(epoch) % 10 == 0:
            print("*********************** Epoch",epoch + 1, "Over **********************")
            print(" ")
            print(" ")
        
        if loss < 0.74:
            break
```

## 5. Visualization of the Cross Entropy Loss Function


```python
plt.figure(figsize = (14, 6))
length = loss_list.__len__()
print("The length of loss_list is:", length) 
plt.plot(np.arange (1, length + 1, 1), loss_list, "black") 
plt.xlabel("epoch") 
plt.ylabel ("loss") 
plt.show()
```

## 6. Prediction on the Test Set and Model Evaluation


```python
data_test.__len__()
```


```python
plt.imshow(data_test[0])
print("The label of this image is:", label_test[0])
```


```python
pred_vec = DNN_Model.forward(data_test[0]. reshape(1, -1))
print("Prediction for data_test[0]:")
Matrix(pred_vec.detach().numpy())
```


```python
proba_distribution = F.softmax(DNN_Model.forward(data_test[0].reshape(1, -1)), dim = 1)
print("Probability distribution for the prediction of data_test[0]:")
print("    ->The argmax of the probability distribution vector is:", 
      torch.argmax(proba_distribution).detach().numpy())
print("    ->Sum of the probability distribution vector is:", torch.sum(proba_distribution).detach().numpy())
Matrix(proba_distribution.detach().numpy())
```


```python
pred = []
for i in range(data_test.__len__()):
    temp_distribution = F.softmax(DNN_Model.forward(data_test[i].reshape(1, -1)), dim = 1)
    pred.append(torch.argmax(temp_distribution).detach().numpy ())
```


```python
print("The accuracy score is:", 100.0 * accuracy_score(label_test, pred), "%")
```
