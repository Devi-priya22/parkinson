import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def mach():
    df=pd.read_csv('park.data')
    df.head()
    features=df.loc[:,df.columns!='status'].values[:,1:]
    labels=df.loc[:,'status'].values
    print(labels[labels==1].shape[0], labels[labels==0].shape[0])
    scaler=MinMaxScaler((-1,1))
    x=scaler.fit_transform(features)
    y=labels
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
    model=XGBClassifier()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(accuracy_score(y_test, y_pred)*100)
    input_data=(119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)


    ## changing input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    ## reshape the numpy array
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    ## standarize the data or reshape the data
    std_data= scaler.transform(input_data_reshaped)

    prediction= model.predict(std_data)
    print(prediction)

    if(prediction[0]==0):
        return False
    else:
        return True