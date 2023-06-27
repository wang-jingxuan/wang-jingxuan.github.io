
# Pytorch DNN Stategy

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
import sympy
from sympy import Matrix
import tushare as ts

%config InlineBackend. figure_format = "svg" # In order to make the figures clearly shown in the notebook
```


```python
import tensorflow as tf
from tensorflow import keras
```


```python
df = ts.lpr_ma_data() #取当前年份的数据
#df = ts.lpr_ma_data(2014) #取2014年的数据
```


```python
df
```

## 2. Data Generating


```python
data = ts.get_k_data('hs300', start = '2014-07-01', end = '2017-03-20')
data.set_index("date", inplace = True)
data
```

    本接口即将停止更新，请尽快使用Pro版接口：https://tushare.pro/document/2



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /var/folders/j0/k9_1_n411h3ct9_rf7__b9hh0000gn/T/ipykernel_7429/3567945227.py in ?()
    ----> 1 data = ts.get_k_data('hs300', start = '2014-07-01', end = '2017-03-20')
          2 data.set_index("date", inplace = True)
          3 data


    /opt/anaconda3/lib/python3.9/site-packages/tushare/stock/trading.py in ?(code, start, end, ktype, autype, index, retry_count, pause)
        702     else:
        703         raise TypeError('ktype input error.')
        704     data = pd.DataFrame()
        705     for url in urls:
    --> 706         data = data.append(_get_k_data(url, dataflag, 
        707                                        symbol, code,
        708                                        index, ktype,
        709                                        retry_count, pause), 


    /opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py in ?(self, name)
       5985             and name not in self._accessors
       5986             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       5987         ):
       5988             return self[name]
    -> 5989         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'append'



```python
data_cleaned = pd.DataFrame()
data_cleaned["Close0"] = np.log(data["close"])
data_cleaned["H_L"] = data["high"] - data["low"]
data_cleaned["C_0"] = data["close"] - data ["open"]
data_cleaned["volume"] = np.log10(data["volume"])

data_cleaned["Closel"] = data_cleaned["Close0"].shift(1)
data_cleaned["Close2"] = data_cleaned["Close0"].shift(2)
data_cleaned["Close3"] = data_cleaned["Close0"].shift(3)
data_cleaned["Close4"] = data_cleaned["Close0"].shift(4)
data_cleaned["Close5"] = data_cleaned["Close0"].shift(5)

data_cleaned["H_L1"] = data_cleaned["H_L"].shift(1)
data_cleaned["H_L2"] = data_cleaned["H_L"].shift(2)
data_cleaned["H_L3"] = data_cleaned["H_L"].shift(3)
data_cleaned["H_L4"] = data_cleaned["H_L"].shift(4)
data_cleaned["H_L5"] = data_cleaned["H_L"].shift(5)

data_cleaned["C_01"] = data_cleaned["C_0"].shift(1)
data_cleaned["C_02"] = data_cleaned["C_0"].shift(2)
data_cleaned["C_03"] = data_cleaned["C_0"].shift(3)
data_cleaned["C_04"] = data_cleaned["C_0"].shift(4)
data_cleaned["C_05"] = data_cleaned["C_0"].shift(5)

data_cleaned["volumel"] = data_cleaned["volume"].shift(1)
data_cleaned["volume2"] = data_cleaned["volume"].shift(2)
data_cleaned["volume3"] = data_cleaned["volume"].shift(3)
data_cleaned["volume4"] = data_cleaned["volume"].shift(4)
data_cleaned["volume5"] = data_cleaned["volume"].shift(5)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 2
          1 data_cleaned = pd.DataFrame()
    ----> 2 data_cleaned["Close0"] = np.log(data["close"])
          3 data_cleaned["H_L"] = data["high"] - data["low"]
          4 data_cleaned["C_0"] = data["close"] - data ["open"]


    NameError: name 'data' is not defined



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

### Remarks

**Shortcomings of DNN strategy**

- Cannot represent all the historical data in the model.
  
  Add long data window? Maybe it will work, but we have better instruments.
- RNN: GRU/ LSTM

**Shortcomings of the feature engineering**

- Only contains price data: technical analysis

- Many data can be further introduced
  
  Fundamental data
  
  text data: (NLP) sentiment analysis, sequence models(RNN: GRU/ LSTM, BRNN)
  
  Many practitioners in China are finding good factors according to a wealth of data available to them.

***
**Shortcomings of the idea**

- Frequently trading will cause large trading cost

- No risk management

- Only trade one asset(HS300), cannot be applied to portfolio management

- Short selling

***

**Other considerations**

- Write all your strategies into a class

- Most code in the financial industry is OOP

- My investment philosophy: fundamental or event-driven(NLP and sentiment analysis)


```python

```
