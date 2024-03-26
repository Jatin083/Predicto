import streamlit as st
from datetime import date 

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from user_upload import upload_and_predict



start = '2015-01-01'
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stock = ("AAPL", "GOOG", "MSFT","GME")
Select_stock = st.selectbox("select dataset for prediction", stock)

n_years = st.slider("Year of prediction:", 1 ,10)
period = n_years*365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker,start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(Select_stock)
data_load_state.text("Loading data...done")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))  
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True) 
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds","Close":"y"})  

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)                

st.subheader('Forecast data')
st.write(forecast.tail())

st.subheader('Forecast data')  
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader('Forecast components')  
fig2 = m.plot_components(forecast)
st.write(fig2)  
upload_and_predict(period)