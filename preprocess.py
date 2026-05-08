import pandas as pd
import numpy as np
import os

# 1. Define Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def preprocess_data():
    print("🚀 Starting Preprocessing...")

    # Load data
    calendar = pd.read_csv(os.path.join(RAW_DATA_PATH, "calendar.csv"))
    sales = pd.read_csv(os.path.join(RAW_DATA_PATH, "sales_train_evaluation.csv"))
    prices = pd.read_csv(os.path.join(RAW_DATA_PATH, "sell_prices.csv"))

    # To save memory, let's take only 50 products (SKUs)
    # You can increase this later once the model works
    sales = sales.head(50)

    print("✅ Files Loaded. Reshaping (Melting) data...")

    # 2. MELT: Convert from Wide (day columns) to Long (rows)
    d_cols = [c for c in sales.columns if 'd_' in c]
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    df = pd.melt(sales, 
                 id_vars=id_cols, 
                 value_vars=d_cols, 
                 var_name='d', 
                 value_name='sales')

    # 3. MERGE: Combine with Calendar and Prices
    df = df.merge(calendar, on='d', how='left')
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    print("✅ Merge complete. Engineering features...")

    # 4. FEATURE ENGINEERING
    # Create continuous time index (Required for TFT)
    df['time_idx'] = df['d'].str.replace('d_', '').astype(int)
    
    # Date features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month.astype(str).astype("category")
    df['day_of_week'] = df['date'].dt.dayofweek.astype(str).astype("category")
    
    # Fill missing prices with average price of that item
    df['sell_price'] = df['sell_price'].fillna(df.groupby('item_id')['sell_price'].transform('mean'))

    # Convert IDs to categories (Better for AI embeddings)
    for col in id_cols:
        df[col] = df[col].astype("category")

    # 5. SAVE: Store the model-ready file
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        
    save_path = os.path.join(PROCESSED_DATA_PATH, "processed_data.pkl")
    df.to_pickle(save_path)
    
    print(f"🎉 Success! Processed data saved to: {save_path}")
    print(f"Total Rows: {len(df)}")

if __name__ == "__main__":
    preprocess_data()