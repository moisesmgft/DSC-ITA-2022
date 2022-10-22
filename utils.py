from platform import platform
from turtle import title
import pandas_datareader as pdr
from datetime import datetime
import json
from os import listdir
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Reshape
from matplotlib import pyplot as plt
from keras import backend

def download_data(tickers_path, dst_path, start_date=datetime(1970, 1, 1), end_date=datetime.now().date(), columns=['Close']):

    assert 'Close' in columns

    with open(tickers_path, 'r') as f:
        tickers = json.load(f)

    success = []
    fail = []

    for ticker in tickers:
        
        ticker = ticker.replace(".", "-")

        try:
            df = pdr.get_data_yahoo(symbols=ticker, start=start_date, end=end_date)

            drop_list = list(set(df.columns) - set(columns))
            df.drop(drop_list,axis=1,inplace=True)


            download_path = dst_path + ticker + '.csv'
            df.to_csv(download_path, index=True)

            success.append(ticker)
        except:
            fail.append(ticker)

    return success, fail

def merge_csv(src_path, dst_path, filename='dataset', how='outer'):
    
    files = [src_path + x for x in listdir(src_path) if 'csv' in x]

    df1 = pd.read_csv(files[0])
    df1 = df1.rename({'Close': files[0].replace(src_path,'').replace('.csv','')}, axis=1)

    for i in range(1,len(files)):
        df2 = pd.read_csv(files[i])
        df2 = df2.rename({'Close': files[i].replace(src_path,'').replace('.csv','')}, axis=1)

        df1 = pd.merge(df1, df2, how=how, on=['Date','Date']) 

    df1['Date'] = pd.to_datetime(df1['Date'])
    df1 = df1.sort_values(by='Date')

    df1 = df1.set_index('Date')
    
    df1.to_csv(dst_path + filename + '.csv')


def dataa(dataframe, data_size, input_tickers, output_tickers, step_size=0, input_size=60, output_size=20, feature_range=(0,1)):

    df = dataframe.copy()
    # predict_df = df[-input_size:]
    df = df[-data_size-input_size:-input_size]
    df = df.dropna(axis=1)

    in_df = df.copy()
    out_df = df.copy()

    in_drop_list = list(set(df.columns) - set(input_tickers))
    out_drop_list = list(set(df.columns) - set(output_tickers))

    print(set(input_tickers) - set(df.columns))

    print(len(in_df.columns), len(out_df.columns))

    in_df.drop(in_drop_list,axis=1,inplace=True)
    out_df.drop(out_drop_list,axis=1,inplace=True)
    # predict_df.drop(in_drop_list,axis=1,inplace=True)

    print(len(in_df.columns), len(out_df.columns))

    scaler = MinMaxScaler(feature_range=feature_range)
    in_dataset_scaled = scaler.fit_transform(in_df.values)
    out_dataset_scaled = scaler.fit_transform(out_df.values)
    # x_predic = scaler.fit_transform(predict_df.values)

    x = []
    y = []

    for i in range(input_size, in_dataset_scaled.shape[0]-output_size, step_size):
        # adicionar passo
        x.append(in_dataset_scaled[i-input_size:i,:])
        y.append(out_dataset_scaled[i:i+output_size,:])

    x, y = np.array(x), np.array(y)

    dic = {ticker:[] for ticker in output_tickers}

    # return scaler, x, y, dic, x_predic
    return scaler, x, y, dic


def train_val_test_split(x, y, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

    return x_train, y_train, x_val, y_val, x_test, y_test


 
def var(y_true, y_pred, axis):
	return backend.mean(backend.sum((y_pred - y_true) - backend.mean(y_pred - y_true, axis=axis, keepdims=True), axis=axis, keepdims=True), axis=axis)

def mse(y_true, y_pred, axis=-1):
    return backend.mean(backend.square(y_pred - y_true), axis=axis)

def E(y_true, y_pred):
	return backend.sqrt(var(y_true, y_pred,axis=0) + mse(y_true, y_pred, axis=0))


def create_model(input_shape, output_shape, layers_info, loss=E, metrics=['accuracy', 'mse']):

    num_els = output_shape[0] * output_shape[1]

    model = Sequential()

    (units, dropout_rate, ret) = layers_info[0]

    model.add(LSTM(units=units,return_sequences=ret, input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    
    for (units, dropout_rate, ret) in layers_info[1:]:

        model.add(LSTM(units=units, return_sequences=ret))
        model.add(Dropout(dropout_rate))

    model.add(Dense(units=num_els))
    model.add(Reshape(output_shape))
    
    model.compile(optimizer='adam',loss=loss,metrics=metrics)

    return model

def validation_plot(train_arr, val_arr, title, f=plt.show, plot_type='loss',):

    # f= lambda: plt.savefig(path)

    if plot_type == 'loss' or plot_type == 'mse':
        argf = np.argmin
        valf = np.min
        title = 'min'
        pos1, pos2 = 'upper right', 'upper left'
        
    else:
        argf = np.argmax
        valf = np.max
        title = 'max'
        pos1, pos2 = 'lower right', 'lower left'

    plt.plot(train_arr)
    plt.plot(val_arr)
    plt.title(title)
    plt.ylabel(plot_type)
    plt.xlabel('epoch')
    legend1 = plt.legend(['train', 'validation'], loc=pos1, title=plot_type)


    xval = argf(val_arr)
    yval = valf(val_arr)

    xtrain = argf(train_arr)
    ytrain = valf(train_arr)

    plt.scatter(xtrain, ytrain)
    plt.scatter(xval, yval)
    
    legend2 = plt.legend([str(ytrain), str(yval)], loc=pos2, title=title)

    ax = plt.gca()
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    f()
    plt.close()

def get_tickers_in_range(df, day_range):

    (min, max) = day_range
    

    df1 = df.copy()
    df1 = df1[- (min - 1):].dropna(axis=1)

    if max == 'MAX':
        return list(df1.columns)

    df2 = df.copy()
    df2 = df2[- (max - 1):].dropna(axis=1)

    tickers_list = list(set(df1.columns) - set(df2.columns))

    return tickers_list

def array_to_dict(dic, arr, tickers):

    d = dic.copy()

    for i in range(arr.shape[1]):
        ticker = tickers[i]
        d[ticker] = arr[0,i,:]

    return d
