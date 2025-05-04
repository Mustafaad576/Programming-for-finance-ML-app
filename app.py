import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data, preprocess_data, feature_engineering, split_data
from models import train_model, evaluate_model
import yfinance as yf

# Streamlit Page Config
st.set_page_config(page_title="Finance ML App", layout="wide")

# Themes (black, orange and purple)
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

# Welcome code
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGRheG9yZ3Zudnp4ZnpvNDBqY292cWt1M2hhejVlZ245ajJydHVwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YRw676NBrmPeM/giphy.gif", use_column_width=True)
st.title("📈 Welcome to Finance ML Explorer")
st.markdown("Upload data or fetch stock info. Then walk through an ML pipeline.")

# Sidebar code
st.sidebar.title("📊 Navigation")
data_option = st.sidebar.selectbox("Choose Data Source", ["Upload Kragle File", "Fetch Yahoo Finance"])
uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset", type=["csv"])
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (Yahoo Finance)")

# Loading Data
if st.button("🔍 Load Data"):
    if data_option == "Upload Kragle File" and uploaded_file:
        df = load_data(uploaded_file)
    elif data_option == "Fetch Yahoo Finance" and stock_symbol:
        df = yf.download(stock_symbol, period="6mo")
    else:
    try:
        df = pd.read_csv("kaggle demo.csv")  
        st.info("No file provided. Loaded demo dataset from repo.")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.warning("Please upload a file or enter a stock symbol.")
        df = None


    if df is not None:
        st.success("Data loaded successfully!")
        st.dataframe(df.head())

