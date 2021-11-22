import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np

st.title("Stock Market Prediction")

wiki = 'https://en.wikipedia.org/wiki/'
tickers = pd.read_html(wiki+'NIFTY_50')[1].Symbol.to_list()
tickers_yf = [i + '.NS' for i in tickers]
stocks = np.array(tickers_yf)

selected_stock = st.selectbox("Select Stock for prediction ", stocks)

def load_data(stock_symbol):
    data = yf.download(tickers=stock_symbol,period='5y',interval='1d')
    data.reset_index(inplace=True)
    return data
if st.button('Select Stock'):
    data_load_state = st.text("Loading Data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading Data... Done!")

    st.subheader("Raw Data")
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Opening Prices'))
        fig.layout.update(title_text='Stock Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    #DATA NORMALIZATION

    from sklearn.preprocessing import MinMaxScaler

    prices = data[['Open']].values
    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    scaled_prices = normalizer.fit_transform(np.array(prices).reshape(-1,1))
    percent = (100*9)/(len(scaled_prices))
    train_size = int(len(scaled_prices)*percent) + 1
    test_size = len(scaled_prices) - train_size

    #Splitting data between train and test
    train_prices, test_prices = scaled_prices[0:train_size,:], scaled_prices[train_size:len(scaled_prices),:1]

    #creating dataset in time series for LSTM model 
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for training
    time_stamp = 100
    x_train, y_train = create_ds(train_prices,time_stamp)
    x_test, y_test = create_ds(test_prices,time_stamp)

    #Reshaping data to fit into LSTM model
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

    #CREATING LSTM MODEL
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    model = Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    #Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=100, verbose=1)

    #Getting the last 'time_stamp' days records
    fut_inp = test_prices[len(test_prices)-time_stamp:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price using the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=time_stamp
    i=0
    while(i<30):

        if(len(tmp_inp)>time_stamp):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    ds_new = scaled_prices.tolist()
    ds_new.extend(lst_output)
    forecast = normalizer.inverse_transform(ds_new)

    length=len(scaled_prices)-1
    st.subheader('Predicted Data')
    st.write(forecast[length:])

    df_final = pd.DataFrame(forecast ,columns = ['price'])
    df_final.reset_index(level=0, inplace=True)
    df_final['index']=df_final['index']-length

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_final['index'], y=df_final['price'], name='Stock Opening Prices'))
    fig2.layout.update(title_text='Predicted Stock Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)
