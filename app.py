import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from io import BytesIO

# App title and theme
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>📈 Stock Price Trend & ML Prediction</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Select Stock & Date Range")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Fetch data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found. Please check the ticker symbol and date range.")
    st.stop()

# Plot raw prices
st.subheader(f"Stock Price Trend for {ticker.upper()}")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers', name='Close Price'))
fig_price.update_layout(title='Stock Closing Price Over Time', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark')
st.plotly_chart(fig_price, use_container_width=True)

# Moving Averages
st.subheader("📊 Moving Averages (7-day & 30-day)")
df['7_MA'] = df['Close'].rolling(window=7).mean()
df['30_MA'] = df['Close'].rolling(window=30).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['7_MA'], mode='lines', name='7-day MA'))
fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['30_MA'], mode='lines', name='30-day MA'))
fig_ma.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark')
st.plotly_chart(fig_ma, use_container_width=True)

# Machine Learning: Predict next 7 days
st.subheader("🤖 Predict Next 7 Days with Linear Regression")

# Prepare data
df_ml = df[['Close']].copy()
df_ml['Prediction'] = df_ml['Close'].shift(-7)

X = df_ml.dropna()[['Close']].values
y = df_ml.dropna()['Prediction'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Train model
model = LinearRegression()
model.fit(X_scaled, y_scaled)

# Predict future
last_close = df_ml[['Close']].values[-1].reshape(1, -1)
last_scaled = scaler.transform(last_close)
predictions_scaled = [model.predict(last_scaled)[0][0]]

for _ in range(6):
    next_scaled = np.array([[predictions_scaled[-1]]])
    pred_scaled = model.predict(next_scaled)[0][0]
    predictions_scaled.append(pred_scaled)

# Inverse scale
predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# Create DataFrame
future_dates = pd.date_range(df['Date'].iloc[-1] + timedelta(days=1), periods=7)
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
st.dataframe(pred_df)

# Plot prediction
st.subheader("📈 Forecast Plot")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close'))
fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted_Close'], mode='lines+markers', name='Predicted Close'))
fig_pred.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark')
st.plotly_chart(fig_pred, use_container_width=True)

# Download full data with predictions
st.subheader("⬇️ Download Full Data with Predictions")
combined_df = pd.concat([df[['Date', 'Close']], pred_df.rename(columns={'Predicted_Close': 'Close'})])
csv = combined_df.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name=f"{ticker}_forecast_data.csv", mime="text/csv")
