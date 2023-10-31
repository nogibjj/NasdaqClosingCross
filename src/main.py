"""
EDA
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_absolute_error


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def LSTM_impimentation(df: pd.DataFrame, n_steps: int=3, target: str='target'):
    target_column = target # column you want to predict
    n_steps = n_steps # number of steps to look back

    # Prepare your data
    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df[i-n_steps:i].values)
        y.append(df.loc[i, target_column])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape X to fit LSTM layer input shape requirements
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df.shape[1]))
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, df.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit model
    model.fit(X_train, y_train, epochs=200, verbose=0)

    # Predict
    preds = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, preds)

    return mae


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df = df.dropna()
    df = df.astype(float)
    print(df.shape)

    unique_id = df['stock_id'].unique()
    count = 0
    mae = 0
    for uid in unique_id:
        count += 2
        data = df[df['stock_id'] == uid]
        # only keep rows that have non-na near_price and far_price
        data_no_nf = data.dropna(subset=['near_price', 'far_price'])
        data_no_nf.reset_index(drop=True, inplace=True)
        # only keep rows that have na near_price and far_price
        data_nf = data[data['near_price'].isna() | data['far_price'].isna()]
        data_nf = data_nf.drop(columns=['near_price', 'far_price'])
        data_nf.reset_index(drop=True, inplace=True)

        mae_no_nf = LSTM_impimentation(data_no_nf)
        
        if data_nf.shape[0] != 0:
            mae_nf = LSTM_impimentation(data_nf)
        else:
            mae_nf = 0
            count -= 1

        mae += mae_no_nf + mae_nf

        print(f'Stock: {uid},  MAE No NF: {mae_no_nf}, MAE NF {mae_nf}, AVG MAE: {mae/count}')

    print(21)
