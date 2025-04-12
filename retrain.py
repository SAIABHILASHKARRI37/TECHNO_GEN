import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import random

# ----------------------------
# Simulate NFT Sales Dataset
# ----------------------------
def simulate_nft_sales_data(n_samples=500):
    collections = [3, 12, 32, 50, 47974]  # encoded
    categories = list(range(10))  # 0 to 9

    data = {
        'collection_encoded': np.random.choice(collections, size=n_samples),
        'asset.num_sales': np.random.randint(1, 15, size=n_samples),
        'category_encoded': np.random.choice(categories, size=n_samples),
        'sale_month': np.random.randint(1, 13, size=n_samples),
        'sale_day': np.random.randint(1, 29, size=n_samples),
        'sale_weekday': np.random.randint(0, 7, size=n_samples)
    }

    df = pd.DataFrame(data)

    # Simulate Ether prices (log scale) based on some weighted influence of features
    df['total_price'] = (
        df['asset.num_sales'] * np.random.uniform(0.01, 0.1, size=n_samples) +
        df['collection_encoded'] * np.random.uniform(0.0001, 0.001, size=n_samples) +
        df['category_encoded'] * np.random.uniform(0.001, 0.005, size=n_samples) +
        np.random.normal(0, 0.1, size=n_samples)
    ).clip(min=0.001)  # always positive

    return df

# ----------------------------
# Model Training Pipeline
# ----------------------------
def train_model(df):
    features = ['collection_encoded', 'asset.num_sales', 'category_encoded', 'sale_month', 'sale_day', 'sale_weekday']
    target = 'total_price'

    X = df[features]
    y = df[target]

    # log1p transform
    y_log = np.log1p(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        tree_method='hist',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"✅ Model trained. RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    return model, features

# ----------------------------
# Save Model & Features
# ----------------------------
def save_artifacts(model, features):
    os.makedirs('app/model', exist_ok=True)
    joblib.dump(model, 'app/model/xgb_nft_model.pkl')
    joblib.dump(features, 'app/model/model_features.pkl')
    print("✅ Model and feature list saved.")

# ----------------------------
# Run Complete Pipeline
# ----------------------------
if _name_ == "_main_":
    df = simulate_nft_sales_data()
    model, features = train_model(df)
    save_artifacts(model, features)