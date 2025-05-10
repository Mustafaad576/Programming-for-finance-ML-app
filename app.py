import streamlit as st
import pandas as pd
import yfinance as yf
import io

from utils import load_data, preprocess_data, feature_engineering, split_data
from models import train_model, evaluate_model

# Streamlit Page Config
st.set_page_config(page_title="Finance ML Predictor", layout="centered")

# Dark Theme CSS (Optional)
st.markdown("""
    <style>
        .stApp {
            background-color: #0d0d0d;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #ff6600;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff4500;
        }
        .stTextInput>input {
            background-color: #333;
            color: white;
            border: 1px solid #ff6600;
        }
    </style>
""", unsafe_allow_html=True)

# Your Linked GIF
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGRheG9yZ3Zudnp4ZnpvNDBqY292cWt1M2hhejVlZ245ajJydHVwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YRw676NBrmPeM/giphy.gif", use_container_width=True)

# Title
st.title("📊 Finance ML Predictor")

# Sidebar options
st.sidebar.header("Select Data Source")
data_option = st.sidebar.selectbox("Choose source:", ["Upload CSV", "Fetch from Yahoo Finance"])
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
stock_symbol = st.sidebar.text_input("Enter Stock Symbol")

df = None

# Load Data
if st.button("📥 Load Data"):
    if data_option == "Upload CSV" and uploaded_file:
        df = load_data(uploaded_file)
    elif data_option == "Fetch from Yahoo Finance" and stock_symbol:
        df = yf.download(stock_symbol, period="6mo")
    else:
        st.warning("Please upload a file or enter a valid stock symbol.")

    if df is not None and not df.empty:
        st.success("✅ Data loaded successfully!")
        st.write(df.head())

        # ML Pipeline
        try:
            df_processed = preprocess_data(df)
            df_features = feature_engineering(df_processed)
            X_train, X_test, y_train, y_test = split_data(df_features)

            model = train_model(X_train, y_train)
            predictions, results_df = evaluate_model(model, X_test, y_test)

            st.success("🎯 Model trained and predictions generated.")
            st.write(results_df.head())

            # Download results
            csv = io.StringIO()
            results_df.to_csv(csv, index=False)
            st.download_button(
                label="📁 Download Predictions",
                data=csv.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Error in ML pipeline: {e}")
    else:
        st.warning("No data loaded. Please check your input.")

