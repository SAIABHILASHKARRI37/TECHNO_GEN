import streamlit as st
import pandas as pd
import numpy as np
import joblib
from live_eth_price import get_live_eth_price_usd

# Load model and features
model = joblib.load('xgb_nft_model.pkl')
features = joblib.load('model_features.pkl')

collection_map = {'DEAD': 12, 'PIXEL': 32, 'poP twtzZ c0': 50, 'dreamscapes': 47974, 'Aesthetic': 3}
category_map = {'Art': 0, 'Collectibles': 1, 'Domain': 2, 'Music': 3, 'Photography': 4,
                'Sports': 5, 'Trading Cards': 6, 'Uncategorized': 7, 'Utility': 8, 'Virtual Worlds': 9}

# Streamlit Page Config
st.set_page_config(page_title="🧠 NFT Price Predictor", page_icon="🎨", layout="centered")

# Custom CSS for dark theme and modern fonts
st.markdown("""
    <style>
    .main {
        background-color: #2e2e2e;
        color: #f1f1f1;
    }
    .stApp {
        background-color: #333;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #008080;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #005f5f;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: #444;
        color: #f1f1f1;
        border: none;
        border-radius: 5px;
    }
    .stSelectbox>div {
        border-radius: 5px;
    }
    .stError {
        color: #f44336;
    }
    .stSuccess {
        color: #4CAF50;
    }
    .stAlert {
        background-color: #333;
        border-left: 5px solid #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 NFT Price Prediction App")
st.markdown("Use this smart tool to *predict NFT prices* based on their metadata & market trends.")

# Input Panel
with st.form("prediction_form"):
    st.subheader("📊 NFT Metadata Input")
    col1, col2 = st.columns(2)

    with col1:
        collection = st.selectbox("🧾 Collection", list(collection_map.keys()))
        category = st.selectbox("🎨 Category", list(category_map.keys()))
        num_sales = st.slider("🔁 Number of Sales", 1, 15, 3)

    with col2:
        month = st.selectbox("📅 Sale Month", list(range(1, 13)))
        day = st.selectbox("📆 Sale Day", list(range(1, 29)))
        weekday = st.selectbox("📍 Weekday (0=Mon)", list(range(7)))

    submit_btn = st.form_submit_button("🚀 Predict Now")

# Prediction Logic
if submit_btn:
    try:
        input_data = pd.DataFrame([{
            'collection_encoded': collection_map[collection],
            'asset.num_sales': num_sales,
            'category_encoded': category_map[category],
            'sale_month': month,
            'sale_day': day,
            'sale_weekday': weekday
        }])

        input_data = input_data[features]
        prediction_log = model.predict(input_data)
        prediction_ether = np.expm1(prediction_log)

        eth_price = get_live_eth_price_usd()
        if eth_price:
            prediction_usd = prediction_ether[0] * eth_price

            st.markdown("---")
            st.subheader("📈 Predicted NFT Price")
            st.success(f"💰 *{prediction_ether[0]:.4f} ETH* ≈ *${prediction_usd:.2f} USD*")
        else:
            st.error("⚠ Unable to fetch live ETH price.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")