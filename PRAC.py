import pandas_datareader.data as web
import datetime
start = datetime.datetime(2020,12,1)
end = datetime.datetime(2021,12,13)
df = web.DataReader('GOOGL', 'stooq', start, end)


def Stock_Price_LSTM_Data_Precesing(df,mem_his_days,pre_days):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    df['label'] = df['Close'].shift(-pre_days)

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

    print(X)
    X_lately = X[-pre_days:]
    X = X[:-pre_days]

    y = df['label'].values[mem_his_days-1:-pre_days]

    import numpy as np
    X = np.array(X)
    y = np.array(y)

    return X,y,X_lately


X,y,X_lately = Stock_Price_LSTM_Data_Precesing(df,5,10)
pre_days = 30
# mem_days = [5,10,15]
# lstm_layers = [1,2,3]
# dense_layers = [1,2,3]
# units = [16,32]
mem_days = [5]
lstm_layers = [1]
dense_layers = [1]
units = [32]
from tensorflow.keras.callbacks import ModelCheckpoint

for the_mem_days in mem_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                filepath = './models/{val_mape:.2f}_{epoch:02d}_' + f'men_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_units}'
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
                              loss='mse',
                              metrics=['mape'])

                model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                          callbacks=[checkpoint])