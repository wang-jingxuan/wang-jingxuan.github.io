

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
```


```python
market_data=pd.read_csv('marketdataprocesssed.csv')
news_data=pd.read_csv('newsdatapreprocessed.csv')
```


```python
market_data=market_data.iloc[:,1:-1]
```


```python
market_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>assetCode</th>
      <th>assetName</th>
      <th>volume</th>
      <th>close</th>
      <th>open</th>
      <th>returnsClosePrevRaw1</th>
      <th>returnsOpenPrevRaw1</th>
      <th>returnsClosePrevMktres1</th>
      <th>returnsOpenPrevMktres1</th>
      <th>returnsClosePrevRaw10</th>
      <th>returnsOpenPrevRaw10</th>
      <th>returnsClosePrevMktres10</th>
      <th>returnsOpenPrevMktres10</th>
      <th>returnsOpenNextMktres10</th>
      <th>universe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-01-02</td>
      <td>A.N</td>
      <td>Agilent Technologies Inc</td>
      <td>3030118.0</td>
      <td>16.24</td>
      <td>15.60</td>
      <td>0.039028</td>
      <td>0.045576</td>
      <td>0.029112</td>
      <td>0.042122</td>
      <td>-0.005511</td>
      <td>-0.037037</td>
      <td>-0.026992</td>
      <td>-0.033293</td>
      <td>0.179633</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-01-02</td>
      <td>AAI.N</td>
      <td>AirTran Holdings Inc</td>
      <td>1551494.0</td>
      <td>4.51</td>
      <td>4.36</td>
      <td>0.015766</td>
      <td>-0.035398</td>
      <td>-0.018756</td>
      <td>-0.047927</td>
      <td>0.127500</td>
      <td>0.141361</td>
      <td>0.110937</td>
      <td>0.144485</td>
      <td>0.048476</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-01-02</td>
      <td>AAP.N</td>
      <td>Advance Auto Parts Inc</td>
      <td>795900.0</td>
      <td>34.14</td>
      <td>33.86</td>
      <td>0.014562</td>
      <td>0.022652</td>
      <td>-0.010692</td>
      <td>0.009156</td>
      <td>0.035283</td>
      <td>0.047398</td>
      <td>-0.005260</td>
      <td>0.054363</td>
      <td>0.029782</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
for columnname in market_data.columns:
    if market_data[columnname].dtype=='float64':
        market_data[columnname]=market_data[columnname].astype('float32')
```


```python
market_data['time'] =  pd.to_datetime(market_data['time'], format='%Y-%m-%d')
```


```python
market_data['assetName']=market_data['assetName'].astype('category')
market_data['assetCode']=market_data['assetCode'].astype('category')
market_data['universe'] = market_data['universe'].astype('bool')
```


```python
news_data.drop(['noveltyCount12H','volumeCounts12H'], axis=1, inplace=True)
```


```python
for columnname in news_data.columns:
    if news_data[columnname].dtype=='float64':
        news_data[columnname]=news_data[columnname].astype('float32')
    elif news_data[columnname].dtype=='int64':
        news_data[columnname]=news_data[columnname].astype('int32')
news_data['urgency']=news_data['urgency'].astype('int8')
news_data['time'] =  pd.to_datetime(news_data['time'], format='%Y-%m-%d')

```


```python
news_data=news_data.iloc[:,1:]
```


```python
def preprocess_news(news_train):
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train

news_data = preprocess_news(news_data)
```


```python
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    return index_df

index_df = unstack_asset_codes(news_data)

def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

news_data = merge_news_on_index(news_data, index_df)
del  index_df

```


```python
news_data.time.max()

```




    Timestamp('2016-12-30 22:00:00')




```python
news_data.time.min()
```




    Timestamp('2009-01-01 00:25:02')




```python
news_data.assetCode.nunique()
```




    13205




```python
news_data.drop_duplicates(subset=['time','assetCode','assetName'],inplace=True)
```


```python
market_data.assetCode.unique()
```




    [A.N, AAI.N, AAP.N, AAPL.O, AB.N, ..., MTGE.O, SITE.N, FCB.N, AMC.N, CVGW.O]
    Length: 3464
    Categories (3464, object): [A.N, AAI.N, AAP.N, AAPL.O, ..., SITE.N, FCB.N, AMC.N, CVGW.O]




```python
news_data.shape
```




    (14271734, 19)




```python
market_data.shape
```




    (3340140, 16)




```python
news_data=news_data[news_data['assetCode'].isin(market_data.assetCode.unique())]
```


```python
df = market_data.merge(news_data, how='left', on=['assetCode', 'time', 'assetName'])
del market_data, news_data

```


```python
df.shape
```




    (3340140, 32)




```python
df_train=df[df.time<=pd.Timestamp(2015,6,1)]
df_test=df[df.time>pd.Timestamp(2015,6,1)]
```


```python
del df
```


```python
np.shape(df_train)[0]*100/3340140
```




    78.51368505511745




```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2622467 entries, 0 to 2622466
    Data columns (total 32 columns):
    time                        datetime64[ns]
    assetCode                   object
    assetName                   object
    volume                      float32
    close                       float32
    open                        float32
    returnsClosePrevRaw1        float32
    returnsOpenPrevRaw1         float32
    returnsClosePrevMktres1     float32
    returnsOpenPrevMktres1      float32
    returnsClosePrevRaw10       float32
    returnsOpenPrevRaw10        float32
    returnsClosePrevMktres10    float32
    returnsOpenPrevMktres10     float32
    returnsOpenNextMktres10     float32
    universe                    bool
    urgency                     float64
    bodySize                    float64
    companyCount                float64
    marketCommentary            object
    sentenceCount               float64
    wordCount                   float64
    relevance                   float32
    sentimentClass              float64
    sentimentNegative           float32
    sentimentNeutral            float32
    sentimentPositive           float32
    sentimentWordCount          float64
    noveltyCount3D              float64
    volumeCounts3D              float64
    delay_time                  object
    headlinelength              float64
    dtypes: bool(1), datetime64[ns](1), float32(16), float64(10), object(4)
    memory usage: 482.7+ MB



```python
df_train.drop(['delay_time'],axis=1,inplace=True)
df_test.drop(['delay_time'],axis=1,inplace=True)
```


```python
nullcols=df_train.returnsClosePrevMktres1.isnull()
```


```python
df_train[['time','assetCode','returnsClosePrevMktres1']][nullcols].head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>assetCode</th>
      <th>returnsClosePrevMktres1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>2009-01-02</td>
      <td>BBND.O</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>739</th>
      <td>2009-01-02</td>
      <td>IIVI.O</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>2009-01-02</td>
      <td>ULTA.O</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1641</th>
      <td>2009-01-05</td>
      <td>AIPC.O</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2009-01-05</td>
      <td>DCOM.O</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3062</th>
      <td>2009-01-05</td>
      <td>ULTA.O</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
def imputation(df):
    df['sentimentClass']=df['sentimentClass'].fillna(0)
    df.urgency=df.urgency.fillna(3)

    for cols in df.columns:
        if(df[cols].dtype=='float64' or df[cols].dtype=='float32'):
            df[cols] = df.groupby("assetCode")[cols].transform(lambda x: x.fillna(x.median()))
    for cols in df.columns:
        if(df[cols].dtype=='float64' or df[cols].dtype=='float32'):
            df[cols][df[cols].isnull()]=df[cols].median()
        elif(df[cols].dtype=='bool' or df[cols].dtype.name=='category'):
            df[cols][df[cols].isnull()]=df[cols].value_counts().argmax()        

    df['marketCommentary'][df['marketCommentary'].isnull()]=df['marketCommentary'].value_counts().argmax()
    return df

```


```python
df_train=imputation(df_train)
```

    C:\Users\chinn\Anaconda3\lib\site-packages\numpy\lib\nanfunctions.py:1019: RuntimeWarning: Mean of empty slice
      return np.nanmean(a, axis, out=out, keepdims=keepdims)
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:12: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
    will be corrected to return the positional maximum in the future.
    Use 'series.values.argmax' to get the position of the maximum now.
      if sys.path[0] == '':
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if sys.path[0] == '':
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
    will be corrected to return the positional maximum in the future.
    Use 'series.values.argmax' to get the position of the maximum now.
      
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      



```python
df_train[['time','assetCode','returnsClosePrevMktres1']][nullcols].head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>assetCode</th>
      <th>returnsClosePrevMktres1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>2009-01-02</td>
      <td>BBND.O</td>
      <td>-0.002200</td>
    </tr>
    <tr>
      <th>739</th>
      <td>2009-01-02</td>
      <td>IIVI.O</td>
      <td>-0.000287</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>2009-01-02</td>
      <td>ULTA.O</td>
      <td>0.000449</td>
    </tr>
    <tr>
      <th>1641</th>
      <td>2009-01-05</td>
      <td>AIPC.O</td>
      <td>0.003477</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2009-01-05</td>
      <td>DCOM.O</td>
      <td>0.001298</td>
    </tr>
    <tr>
      <th>3062</th>
      <td>2009-01-05</td>
      <td>ULTA.O</td>
      <td>0.000449</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.isnull().sum()
```




    time                        0
    assetCode                   0
    assetName                   0
    volume                      0
    close                       0
    open                        0
    returnsClosePrevRaw1        0
    returnsOpenPrevRaw1         0
    returnsClosePrevMktres1     0
    returnsOpenPrevMktres1      0
    returnsClosePrevRaw10       0
    returnsOpenPrevRaw10        0
    returnsClosePrevMktres10    0
    returnsOpenPrevMktres10     0
    returnsOpenNextMktres10     0
    universe                    0
    urgency                     0
    bodySize                    0
    companyCount                0
    marketCommentary            0
    sentenceCount               0
    wordCount                   0
    relevance                   0
    sentimentClass              0
    sentimentNegative           0
    sentimentNeutral            0
    sentimentPositive           0
    sentimentWordCount          0
    noveltyCount3D              0
    volumeCounts3D              0
    headlinelength              0
    dtype: int64




```python
df_test=imputation(df_test)
```

    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:12: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
    will be corrected to return the positional maximum in the future.
    Use 'series.values.argmax' to get the position of the maximum now.
      if sys.path[0] == '':
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if sys.path[0] == '':
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
    will be corrected to return the positional maximum in the future.
    Use 'series.values.argmax' to get the position of the maximum now.
      
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      



```python
return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
```


```python
def generate_features(df,window_size=[3,7,14],shift_size=1):
    grouped=df.groupby('assetCode')
    #----------------------------lag features-------------------------------------------
    for col in return_features:
        for window in window_size:
            df['%s_lag_%s_median'%(col,window)] = grouped[col].shift(shift_size).rolling(window=window).mean()
            df['%s_lag_%s_max'%(col,window)] = grouped[col].shift(shift_size).rolling(window=window).max()
            df['%s_lag_%s_min'%(col,window)] = grouped[col].shift(shift_size).rolling(window=window).min()
   
   # df['betareturn1']=(df['returnsClosePrevRaw1'] - df['returnsClosePrevMktres1']) / (df[returnraw1] - df[returnMktres1]).groupby("time").mean()
    df['closeopentovolume']=(df['close']+df['open'])*df['volume']
    df['meanvolume']=grouped['volume'].mean()
    df['meanclose']=grouped['close'].mean()
    df['stdclose']=grouped['close'].std()
    #-----------------------------time features----------------------------------------------
    df['dayofweek']=df.time.dt.dayofweek
    df['quarter']=df.time.dt.quarter
    df['month']=df.time.dt.month
    df['year']=df.time.dt.year
    #---------------------------quant features---------------------------------------------------
    new_column = grouped.apply(lambda x: x['close'].ewm(span=30).mean())
    df["close_30EMA"] = new_column.reset_index(level=0, drop=True)
    new_column = grouped.apply(lambda x: x['close'].ewm(span=26).mean())
    df["close_26EMA"] = new_column.reset_index(level=0, drop=True)
    new_column = grouped.apply(lambda x: x['close'].ewm(span=12).mean())
    df["close_12EMA"] = new_column.reset_index(level=0, drop=True)
    df['MACD'] = df['close_12EMA'] - df['close_26EMA']
    no_of_std = 2
    #--------------------------bolinger band---------------------------------------
    new_column=grouped['close'].rolling(window=7).mean()
    df['MA_7MA'] =  new_column.reset_index(level=0, drop=True)
    new_column=grouped['close'].rolling(window=7).std()
    df['MA_7MA_std'] =  new_column.reset_index(level=0, drop=True)
    df['MA_7MA_BB_high'] = df['MA_7MA'] + no_of_std * df['MA_7MA_std']
    df['MA_7MA_BB_low'] = df['MA_7MA'] - no_of_std * df['MA_7MA_std']
    return df.fillna(-1)
    
    
```


```python
df_train=generate_features(df_train)
df_test=generate_features(df_test)
```


```python
def RSI(df, column="close", period=14):
    # wilder's RSI
    delta = df.groupby('assetCode')[column].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=period - 1,  adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
    rsi = 100 - 100 / (1 + rUp / rDown)    
    return df.join(rsi.to_frame('RSI'))

```


```python
df_train=RSI(df_train)
df_test=RSI(df_test)
```


```python
def beta(df):
    
    df['raw_median'] = df.groupby('time').returnsOpenPrevRaw10.transform('median')
    df['xy'] = df.returnsOpenPrevRaw10 * df.raw_median

    roll = df.groupby('assetCode').rolling(window=20)

    df['cov_xy'] = (
      (roll.xy.mean() - roll.returnsOpenPrevRaw10.mean() * roll.raw_median.mean()) * 20 / 19
      ).reset_index(0,drop=True)

    df['var_y'] = roll.raw_median.var().reset_index(0,drop=True)
    df['beta'] = (df['cov_xy'] /df['var_y'])
    df['beta'] = df.groupby('assetCode')['beta'].shift(1)
    df.drop(['var_y','xy','raw_median','cov_xy'],axis=1,inplace=True)
    return df.fillna(-1)

```


```python
df_train=beta(df_train)
df_test=beta(df_test)
```


```python
df_train['sin_quarter'] = np.sin(2*np.pi*df_train.quarter/4)
```


```python
df_test['sin_quarter'] = np.sin(2*np.pi*df_test.quarter/4)
```


```python
df_test['sin_dayofweek']=np.sin(2*np.pi*df_test.dayofweek/7)
df_train['sin_dayofweek']=np.sin(2*np.pi*df_train.dayofweek/7)
```


```python
df_test['sin_month']=np.sin(2*np.pi*df_test.month/12)
df_train['sin_month']=np.sin(2*np.pi*df_train.month/12)
```


```python
df_train.columns
```




    Index(['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
           'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
           'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
           'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
           'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
           'returnsOpenNextMktres10', 'universe', 'urgency', 'bodySize',
           'companyCount', 'marketCommentary', 'sentenceCount', 'wordCount',
           'relevance', 'sentimentClass', 'sentimentNegative', 'sentimentNeutral',
           'sentimentPositive', 'sentimentWordCount', 'noveltyCount3D',
           'volumeCounts3D', 'headlinelength',
           'returnsClosePrevMktres10_lag_3_median',
           'returnsClosePrevMktres10_lag_3_max',
           'returnsClosePrevMktres10_lag_3_min',
           'returnsClosePrevMktres10_lag_7_median',
           'returnsClosePrevMktres10_lag_7_max',
           'returnsClosePrevMktres10_lag_7_min',
           'returnsClosePrevMktres10_lag_14_median',
           'returnsClosePrevMktres10_lag_14_max',
           'returnsClosePrevMktres10_lag_14_min',
           'returnsClosePrevRaw10_lag_3_median', 'returnsClosePrevRaw10_lag_3_max',
           'returnsClosePrevRaw10_lag_3_min', 'returnsClosePrevRaw10_lag_7_median',
           'returnsClosePrevRaw10_lag_7_max', 'returnsClosePrevRaw10_lag_7_min',
           'returnsClosePrevRaw10_lag_14_median',
           'returnsClosePrevRaw10_lag_14_max', 'returnsClosePrevRaw10_lag_14_min',
           'open_lag_3_median', 'open_lag_3_max', 'open_lag_3_min',
           'open_lag_7_median', 'open_lag_7_max', 'open_lag_7_min',
           'open_lag_14_median', 'open_lag_14_max', 'open_lag_14_min',
           'close_lag_3_median', 'close_lag_3_max', 'close_lag_3_min',
           'close_lag_7_median', 'close_lag_7_max', 'close_lag_7_min',
           'close_lag_14_median', 'close_lag_14_max', 'close_lag_14_min',
           'closeopentovolume', 'meanvolume', 'meanclose', 'stdclose', 'dayofweek',
           'quarter', 'month', 'year', 'close_30EMA', 'close_26EMA', 'close_12EMA',
           'MACD', 'MA_7MA', 'MA_7MA_std', 'MA_7MA_BB_high', 'MA_7MA_BB_low',
           'RSI', 'beta', 'sin_quarter', 'sin_dayofweek', 'sin_month'],
          dtype='object')




```python
num_cols=['volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsClosePrevMktres10_lag_3_median',
       'returnsClosePrevMktres10_lag_3_max',
       'returnsClosePrevMktres10_lag_3_min',
       'returnsClosePrevMktres10_lag_7_median',
       'returnsClosePrevMktres10_lag_7_max',
       'returnsClosePrevMktres10_lag_7_min',
       'returnsClosePrevMktres10_lag_14_median',
       'returnsClosePrevMktres10_lag_14_max',
       'returnsClosePrevMktres10_lag_14_min',
       'returnsClosePrevRaw10_lag_3_median', 'returnsClosePrevRaw10_lag_3_max',
       'returnsClosePrevRaw10_lag_3_min', 'returnsClosePrevRaw10_lag_7_median',
       'returnsClosePrevRaw10_lag_7_max', 'returnsClosePrevRaw10_lag_7_min',
       'returnsClosePrevRaw10_lag_14_median',
       'returnsClosePrevRaw10_lag_14_max', 'returnsClosePrevRaw10_lag_14_min',
       'open_lag_3_median', 'open_lag_3_max', 'open_lag_3_min',
       'open_lag_7_median', 'open_lag_7_max', 'open_lag_7_min',
       'open_lag_14_median', 'open_lag_14_max', 'open_lag_14_min',
       'close_lag_3_median', 'close_lag_3_max', 'close_lag_3_min',
       'close_lag_7_median', 'close_lag_7_max', 'close_lag_7_min',
       'close_lag_14_median', 'close_lag_14_max', 'close_lag_14_min',
       'closeopentovolume', 'meanvolume', 'meanclose', 'stdclose', 'close_30EMA', 'close_26EMA', 'close_12EMA',
       'MACD', 'MA_7MA', 'MA_7MA_std', 'MA_7MA_BB_high', 'MA_7MA_BB_low',
       'RSI', 'beta', 'sin_quarter', 'sin_dayofweek', 'sin_month']
```


```python
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_train[num_cols]=scaler.fit_transform(df_train[num_cols])
df_test[num_cols]=scaler.transform(df_test[num_cols])
```

    C:\Users\chinn\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:625: DataConversionWarning: Data with input dtype float32, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    C:\Users\chinn\Anaconda3\lib\site-packages\sklearn\base.py:462: DataConversionWarning: Data with input dtype float32, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    C:\Users\chinn\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: DataConversionWarning: Data with input dtype float32, float64 were all converted to float64 by StandardScaler.
      after removing the cwd from sys.path.



```python
df_train.to_csv('train.csv')
df_test.to_csv('test.csv')
```
