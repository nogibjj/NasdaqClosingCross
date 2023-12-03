"""
EDA
"""
import os
import concurrent.futures
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def sanity_add(aa_, bb_):
    """sanity check"""
    return aa_ + bb_


def lstm_impimentation(df: pd.DataFrame, n_steps_: int = 3, target: str = "target"):
    """lstm implementation for closing cross target prediction"""
    target_column = target  # column you want to predict
    n_steps = n_steps_  # number of steps to look back

    # Prepare your data
    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df[i - n_steps : i].values)
        y.append(df.loc[i, target_column])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reshape X to fit LSTM layer input shape requirements
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df.shape[1]))
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, df.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Fit model
    model.fit(X_train, y_train, epochs=200, verbose=0)

    # Predict
    preds = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, preds)

    return mae


def process_uid(uid):
    """concurrent process for uid"""
    data = df_[df_["stock_id"] == uid]

    # data_no_nf = data.dropna(subset=["near_price", "far_price"])
    # data_no_nf = data_no_nf.dropna()
    data_no_nf = data.dropna()
    data_no_nf.reset_index(drop=True, inplace=True)

    data_nf = data[data["near_price"].isna() | data["far_price"].isna()]
    data_nf = data_nf.drop(columns=["near_price", "far_price"])
    data_nf = data_nf.dropna()
    data_nf.reset_index(drop=True, inplace=True)

    MAE_NO_NF_ = lstm_impimentation(data_no_nf)

    COUNT_ = 1
    if data_nf.shape[0] != 0:
        MAE_NF = lstm_impimentation(data_nf)
        COUNT_ += 1
    else:
        MAE_NF = 0

    print(f"Stock: {uid},  MAE No NF: {MAE_NO_NF_}, MAE NF {MAE_NF}")

    return MAE_NO_NF_, MAE_NF, COUNT_


if __name__ == "__main__":
    df_ = pd.read_csv("data/train.csv")
    df_ = df_.dropna()
    df_ = df_.astype(float)
    print(df_.shape)

    unique_id = df_["stock_id"].unique()
    COUNT = 0
    MAE_ = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_uid, unique_id))

    # MAE_ = sum(mae_no_nf + MAE_NF_ for mae_no_nf, MAE_NF_, _ in results)
    # COUNT = sum(count for _, _, count in results)
    MAE_ = 0
    COUNT = 0
    df_temp = pd.DataFrame()
    for ii, (MAE_NO_NF, MAE_NF_, count) in enumerate(results):
        MAE_ += MAE_NO_NF + MAE_NF_
        COUNT += count

        # save to csv
        df_temp = df_temp.append(
            {"stock_id": ii, "MAE_NO_NF": MAE_NO_NF, "MAE_NF": MAE_NF_},
            ignore_index=True,
        )

    avg_mae = MAE_ / COUNT
    print(f"AVG MAE: {avg_mae}")
    df_temp.to_csv("data/MAE_{}.csv", index=False)

    # for uid in unique_id:
    #     COUNT += 2
    #     data = df_[df_["stock_id"] == uid]
    #     # only keep rows that have non-na near_price and far_price
    #     data_no_nf = data.dropna(subset=["near_price", "far_price"])
    #     data_no_nf.reset_index(drop=True, inplace=True)
    #     # only keep rows that have na near_price and far_price
    #     data_nf = data[data["near_price"].isna() | data["far_price"].isna()]
    #     data_nf = data_nf.drop(columns=["near_price", "far_price"])
    #     data_nf.reset_index(drop=True, inplace=True)

    #     mae_no_nf = lstm_impimentation(data_no_nf)

    #     if data_nf.shape[0] != 0:
    #         MAE_NF = lstm_impimentation(data_nf)
    #     else:
    #         MAE_NF = 0
    #         COUNT -= 1

    #     MAE_ += mae_no_nf + MAE_NF

    #     print(
    #         f"Stock: {uid},  MAE No NF: {mae_no_nf}, MAE NF {MAE_NF}, AVG MAE: {MAE_/COUNT}"
    #     )

    print(21)
