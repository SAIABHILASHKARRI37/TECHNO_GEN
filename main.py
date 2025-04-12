from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from live_eth_price import get_live_eth_price_usd

app = Flask(__name__)
model = joblib.load('xgb_nft_model.pkl')
features = joblib.load('model_features.pkl')

collection_map = {'DEAD': 12, 'PIXEL': 32, 'poP twtzZ c0': 50, 'dreamscapes': 47974, 'Aesthetic': 3}
category_map = {'Art': 0, 'Collectibles': 1, 'Domain': 2, 'Music': 3, 'Photography': 4, 'Sports': 5,
                'Trading Cards': 6, 'Uncategorized': 7, 'Utility': 8, 'Virtual Worlds': 9}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        df = pd.DataFrame([data])
        df['collection_encoded'] = df['collection_encoded'].map(collection_map)
        df['category_encoded'] = df['category_encoded'].map(category_map)
        df = df[features]

        prediction_log = model.predict(df)
        prediction_ether = np.expm1(prediction_log)

        eth_price = get_live_eth_price_usd()
        if eth_price:
            prediction_usd = float(prediction_ether[0]) * eth_price
            return jsonify({'ether': round(prediction_ether[0], 4), 'usd': round(prediction_usd, 2)})
        else:
            return jsonify({'error': 'Failed to fetch ETH price'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)