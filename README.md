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

## 🧪 Sample Prediction & Encoding Explanation

We performed predictions on new NFT samples using the trained XGBoost model. Here's a breakdown of how encoding was handled and an example prediction:

### 🔢 Collection Encoding (Examples)
- 12: 'DEAD'
- 32: 'PIXEL'
- 50: 'poP twtzZ c0'
- 47974: 'dreamscapes'
- 3: 'Aesthetic'

### 🎨 Category Encoding
- 0: Art  
- 1: Collectibles  
- 2: Domain  
- 3: Music  
- 4: Photography  
- 5: Sports  
- 6: Trading Cards  
- 7: Uncategorized  
- 8: Utility  
- 9: Virtual Worlds  

### 📥 Example Input for Prediction
```python
{
    'collection_encoded': 12,        # Corresponds to 'DEAD'
    'asset.num_sales': 12,           
    'category_encoded': 3,           # Corresponds to 'Music'
    'sale_month': 4,                 
    'sale_day': 10,                  
    'sale_weekday': 3                # 0 = Monday, 6 = Sunday
}

# Predicted Total Price (Ether): 0.7470

"C:\Users\saiab\OneDrive\Pictures\Screenshots\Screenshot 2025-04-13 063932.png"




