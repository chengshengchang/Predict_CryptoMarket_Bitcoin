import time
import requests
import pandas as pd


#取得目前日期與時間 (Unix時間戳記格式)
end_date = int(time.time())




from_thisday = "2021-06-01 00:00:00" # 時間格式為字串
struct_time = time.strptime(from_thisday, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
start_date = int(time.mktime(struct_time)) # 轉成時間戳

period = str(24*60*60)
currency='USDT_BTC'

res = requests.get(('https://poloniex.com/public?command=returnChartData&currencyPair={0}&start={1}&end={2}&period={3}').format(currency,start_date,end_date,period))

df = pd.DataFrame(res.json())
df['date']=pd.to_datetime(df['date'],unit='s')
df = df.drop(columns=['quoteVolume','weightedAverage'])

df.set_index('date',inplace=True)

df = df.reindex(columns=['open','high','low','close','volume'])



def Stock_Price_LSTM_Data_Precesing(df,mem_his_days,pre_days):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    df['label'] = df['close'].shift(-pre_days)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(df.iloc[:,:-1])

    from collections import deque
    deq = deque(maxlen=mem_his_days)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq)==mem_his_days:
            X.append(list(deq))

    X_lately = X[-pre_days:]
    X = X[:-pre_days]

    y = df['label'].values[mem_his_days-1:-pre_days]

    import numpy as np
    X = np.array(X)
    y = np.array(y)

    return X,y,X_lately



X,y,X_lately = Stock_Price_LSTM_Data_Precesing(df,5,10)


pre_days = 3

mem_days = [5]
lstm_layers = [1]
dense_layers = [1]
units = [32]

from tensorflow.keras.callbacks import ModelCheckpoint

for the_mem_days in mem_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                filepath = './models/{val_mape:.2f}_{epoch:02d}_' + f'men_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_units}_MAPE'
                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    save_weights_only=False,
                    monitor='val_mape',
                    mode='min',
                    save_best_only=True)

                X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout

                model = Sequential()
                model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))
                model.add(Dropout(0.1))

                for i in range(the_lstm_layers):
                    model.add(LSTM(the_units, activation='relu', return_sequences=True))
                    model.add(Dropout(0.1))

                model.add(LSTM(the_units, activation='relu'))
                model.add(Dropout(0.1))

                for i in range(the_dense_layers):
                    model.add(Dense(the_units, activation='relu'))
                    model.add(Dropout(0.1))

                model.add(Dense(1))

                model.compile(optimizer='adam',
                              loss='cosine_similarity',
                              metrics=['mape'])

                model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test),
                          callbacks=[checkpoint])