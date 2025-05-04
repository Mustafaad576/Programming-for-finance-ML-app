import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

# Streamlit Page Config
st.set_page_config(page_title="Finance ML App", layout="wide")

# Welcome code
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGRheG9yZ3Zudnp4ZnpvNDBqY292cWt1M2hhejVlZ245ajJydHVwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YRw676NBrmPeM/giphy.gif", use_column_width=True)
st.title("📈 Welcome to Finance ML Explorer")
st.markdown("Upload data or fetch stock info. Then walk through an ML pipeline.")

# Sidebar code
st.sidebar.title("📊 Navigation")
data_option = st.sidebar.selectbox("Choose Data Source", ["Upload Kragle File", "Fetch Yahoo Finance"])
uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset", type=["csv"])
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (Yahoo Finance)")

# Load Data
if st.button("🔍 Load Data"):
    if data_option == "Upload Kragle File" and uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif data_option == "Fetch Yahoo Finance" and stock_symbol:
        df = yf.download(stock_symbol, period="6mo")
    else:
        st.warning("Please upload a file or enter a stock symbol.")
        df = None

    if df is not None:
        st.success("Data loaded successfully!")
        st.dataframe(df.head())

        # Stock Price Trend - Line Chart
        st.subheader("📉 Stock Price Trend")
        fig = px.line(df, x=df.index, y="Close", title="Stock Price Trend")
        st.plotly_chart(fig)

        # Moving Averages (7-day and 30-day)
        st.subheader("📊 Moving Averages (7-day & 30-day)")
        df['7_day_MA'] = df['Close'].rolling(window=7).mean()
        df['30_day_MA'] = df['Close'].rolling(window=30).mean()
        fig = px.line(df, x=df.index, y=["Close", "7_day_MA", "30_day_MA"], title="Stock Price & Moving Averages")
        st.plotly_chart(fig)

        # Price Change Over Time
        st.subheader("📈 Price Change Over Time")
        df['Price Change'] = df['Close'].pct_change() * 100
        fig = px.line(df, x=df.index, y="Price Change", title="Price Change Over Time (%)")
        st.plotly_chart(fig)

        # Scatter Plot for Actual vs Predicted
        st.subheader("📉 Actual vs Predicted Price Change")
        model = LinearRegression()
        df.dropna(subset=['Price Change'], inplace=True) 
        X = np.array(range(len(df))).reshape(-1, 1) 
        y = df['Price Change'].values
        model.fit(X, y)
        y_pred = model.predict(X)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=y, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=df.index, y=y_pred, mode='lines', name='Predicted'))
        fig.update_layout(title="Actual vs Predicted Price Change", xaxis_title="Date", yaxis_title="Price Change (%)")
        st.plotly_chart(fig)

        # Model Evaluation
        st.subheader("📊 Model Evaluation")
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R2 Score: {r2}")

        # Download Button for Model Predictions
        predictions_df = pd.DataFrame({"Date": df.index, "Actual": y, "Predicted": y_pred})
        st.download_button(label="Download Model Predictions", data=predictions_df.to_csv(), file_name="model_predictions.csv", mime="text/csv")
