# TECHNO_GEN

# 🧠 Smart NFT Pricing Model

## 🔍 Project Overview
This project aims to develop a **Smart NFT Pricing Model** that predicts the **market value of NFTs (in Ether)** based on their metadata using machine learning. Using a Kaggle dataset and XGBoost, we built a model that learns patterns from past NFT sales to predict future pricing more accurately.

---

## 📁 Dataset
- **Source**: Kaggle
- **Contents**: NFT metadata such as collection name, category, number of sales, sale date, and total price in Ether
- **Goal**: Use this data to train a regression model to predict prices

---

## 🧹 Data Preprocessing & Feature Engineering
- Removed unnecessary or redundant columns
- Applied `LabelEncoder` to encode categorical features:
  - `collection_encoded`
  - `category_encoded`
- Extracted features from sale date:
  - `sale_month`
  - `sale_day`
  - `sale_weekday`
- Applied `log1p` transformation on the target column (`last_sale.total_price`) to normalize the distribution

---

## 📊 Exploratory Data Analysis
- Selected features for modeling:
  - `collection_encoded`
  - `asset.num_sales`
  - `category_encoded`
  - `sale_month`
  - `sale_day`
  - `sale_weekday`
- Generated a correlation matrix and visualized it using a heatmap to understand feature relationships

---

## 🤖 Model Building
- Performed model selection using **k-fold cross-validation**
- Tried multiple models, and found **XGBoost** to perform best based on evaluation metrics
- Model performance on test set:
  - **RMSE**: 1.2678 Ether
  - **MAE**: 0.4057 Ether

---


