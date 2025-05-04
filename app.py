import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from utils import load_data, preprocess_data, feature_engineering, split_data
from models import train_model, evaluate_model
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="Finance ML App", layout="wide")

# Themes
st.markdown("""
    <style>
        .stApp {
            background-color: #0d0d0d;
        }
        .main > div {
            padding: 2rem;
        }
        .block-container {
            background-color: #1e1e2f;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #ff6600;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Welcome
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGRheG9yZ3Zudnp4ZnpvNDBqY292cWt1M2hhejVlZ245ajJydHVwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YRw676NBrmPeM/giphy.gif", use_column_width=True)
st.title("📈 Welcome to Finance ML Explorer")
st.markdown("Upload data or fetch stock info. Then walk through an ML pipeline.")

# Sidebar
st.sidebar.title("📊 Navigation")
data_option = st.sidebar.selectbox("Choose Data Source", ["Upload Kragle File", "Fetch Yahoo Finance"])
uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset", type=["csv"])
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (Yahoo Finance)")

# Load Data
if st.button("🔍 Load Data"):
    if data_option == "Upload Kragle File" and uploaded_file:
        df = load_data(uploaded_file)
    elif data_option == "Fetch Yahoo Finance" and stock_symbol:
        df = yf.download(stock_symbol, period="6mo")
    else:
        st.warning("Please upload a file or enter a stock symbol.")
        df = None

    if df is not None:
        st.success("Data loaded successfully!")
        st.dataframe(df.head())

        # Stock Price Trend
        st.subheader("Stock Price Trend (Close)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Close Price',
            line=dict(color='orange', width=2)
        ))
        fig.update_layout(title=f"Stock Price Trend for {stock_symbol}",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f',
                          font_color="white")
        st.plotly_chart(fig)

        # Moving Averages
        st.subheader("Moving Averages (7-day & 30-day)")
        df['7_day_MA'] = df['Close'].rolling(window=7).mean()
        df['30_day_MA'] = df['Close'].rolling(window=30).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Close Price',
            line=dict(color='orange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['7_day_MA'], mode='lines', name='7-day MA',
            line=dict(color='purple', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['30_day_MA'], mode='lines', name='30-day MA',
            line=dict(color='white', width=2, dash='dash')
        ))
        fig.update_layout(title=f"Moving Averages for {stock_symbol}",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f',
                          font_color="white")
        st.plotly_chart(fig)

        # Feature Engineering: Price Change
        df['Price Change'] = df['Close'].pct_change()
        df['Price Change'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['Price Change'], inplace=True)

        st.subheader("Price Change Over Time")
        if 'Price Change' in df.columns:
            df_clean = df[['Price Change']].dropna()
            fig = px.line(df_clean, x=df_clean.index, y='Price Change', title=f"Price Change for {stock_symbol}")
            fig.update_layout(plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f', font_color="white")
            st.plotly_chart(fig)
        else:
            st.warning("'Price Change' column not found in DataFrame.")

        # Model Evaluation (ML)
        st.subheader("Model Evaluation (Example with Linear Regression)")
        model, X_test, y_test, y_pred = train_model(df)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R2 Score: {r2}")

        # Predictions Download
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        predictions_df.to_csv('model_predictions.csv', index=False)
        st.download_button('Download Model Predictions', 'model_predictions.csv')
