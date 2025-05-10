import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from utils import load_data, preprocess_data, feature_engineering, split_data
from models import train_model, evaluate_model
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="Finance ML App", layout="wide")

# Theme (black, orange, purple)
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

# Welcome Banner
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGRheG9yZ3Zudnp4ZnpvNDBqY292cWt1M2hhejVlZ245ajJydHVwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YRw676NBrmPeM/giphy.gif", use_column_width=True)
st.title("📈 Welcome to Finance ML Explorer")
st.markdown("Upload data or fetch stock info. Then walk through an ML pipeline.")

# Sidebar controls
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

        # Chart: Stock Price Trend
        st.subheader("Stock Price Trend (Close)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"Stock Price Trend for {stock_symbol}",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f', font_color="white")
        st.plotly_chart(fig)

        # Chart: Moving Averages
        st.subheader("Moving Averages (7-day & 30-day)")
        df['7_day_MA'] = df['Close'].rolling(window=7).mean()
        df['30_day_MA'] = df['Close'].rolling(window=30).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['7_day_MA'], mode='lines', name='7-day MA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['30_day_MA'], mode='lines', name='30-day MA'))
        fig.update_layout(title=f"Moving Averages for {stock_symbol}",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f', font_color="white")
        st.plotly_chart(fig)

        # Feature Engineering - Price Change
        df['Price Change'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        st.subheader("Price Change Over Time")
        if 'Price Change' in df.columns:
            fig = px.line(df, x=df.index, y='Price Change', title=f"Price Change for {stock_symbol}")
            fig.update_layout(plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f', font_color="white")
            st.plotly_chart(fig)
        else:
            st.warning("Price Change column not found!")

        # Train ML Model
        st.subheader("Model Evaluation (Example with Linear Regression)")
        model, X_test, y_test, y_pred = train_model(df)  # Assumes model is trained using 'Price Change'
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R2 Score: {r2}")

        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted Price Change")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.loc[y_test.index].index, y=y_test, mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=df.loc[y_test.index].index, y=y_pred, mode='markers', name='Predicted'))
        fig.update_layout(title="Actual vs Predicted Price Change",
                          xaxis_title="Date", yaxis_title="Price Change",
                          plot_bgcolor='#1e1e2f', paper_bgcolor='#1e1e2f', font_color="white")
        st.plotly_chart(fig)

        # Download predictions as CSV
        st.subheader("📥 Download Machine Learning Predictions")
        result_df = df.loc[y_test.index].copy()
        result_df['Actual Price Change'] = y_test.values
        result_df['Predicted Price Change'] = y_pred
        result_df = result_df[['Close', 'Actual Price Change', 'Predicted Price Change']]
        if result_df.index.name == 'Date' or isinstance(result_df.index, pd.DatetimeIndex):
            result_df.reset_index(inplace=True)

        st.download_button(
            label="Download ML Predictions CSV",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name='model_predictions.csv',
            mime='text/csv'
        )
