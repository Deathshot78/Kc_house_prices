import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def full_preprocess_pipeline(input_csv_path: str, output_csv_path: str):
    """
    Loads raw data, applies all preprocessing steps, and saves the result.
    """
    print(f"Starting preprocessing of {input_csv_path}...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {input_csv_path}")
        return None
    
    if 'id' in df.columns: # Assuming you want to drop 'id'
        df = df.drop('id', axis=1)

    # --- 2. Date Feature Engineering ---
    df['date'] = pd.to_datetime(df['date'])
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df['sale_dayofyear'] = df['date'].dt.dayofyear
    df = df.drop('date', axis=1)
    
    # --- 3. Age and Renovation Features ---
    df['age_at_sale'] = df['sale_year'] - df['yr_built']
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['yrs_since_renovation'] = np.where(df['was_renovated'] == 1, 
                                          df['sale_year'] - df['yr_renovated'], 0)
    df['effective_age'] = df['sale_year'] - np.maximum(df['yr_built'], df['yr_renovated'].fillna(0))

    # --- 4. Boolean/Logical Features ---
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)

    # --- 5. Ratio Features ---
    df['sqft_living_per_bedroom'] = np.where(df['bedrooms'] > 0, df['sqft_living'] / df['bedrooms'], df['sqft_living'])
    df['bathrooms_per_bedroom'] = np.where(df['bedrooms'] > 0, df['bathrooms'] / df['bedrooms'], df['bathrooms'])
    df['sqft_living_per_floor'] = np.where(df['floors'] > 0, df['sqft_living'] / df['floors'], df['sqft_living'])
    df.replace([np.inf, -np.inf], 0, inplace=True) # Handle potential infs

    # --- 6. Target Variable Transformation (CRUCIAL) ---
    # This assumes 'price' is the original price column
    df['price_log'] = np.log1p(df['price']) 

    # --- 7. Categorical Feature Encoding ('zipcode') ---
    df = pd.get_dummies(df, columns=['zipcode'], prefix='zip', dtype=int)

    # --- 8. Numerical Feature Scaling ---
    # Identify features to scale: all numerical columns except booleans, one-hot encoded, and the raw price/log_price target
    features_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    zip_cols = [col for col in df.columns if col.startswith('zip_')]
    boolean_like_cols = ['waterfront', 'was_renovated', 'has_basement'] # Add any other custom booleans
    # Exclude original price, the log-transformed target, and already encoded/boolean features
    cols_to_exclude_from_scaling = target_cols = ['price', 'price_log'] + zip_cols + boolean_like_cols
    
    final_features_to_scale = [col for col in features_to_scale if col not in cols_to_exclude_from_scaling]
    
    if final_features_to_scale:
        scaler = StandardScaler()
        df[final_features_to_scale] = scaler.fit_transform(df[final_features_to_scale])
    
    # --- 9. Save Preprocessed Data ---
    output_directory = os.path.dirname(output_csv_path)
    if output_directory: # Ensure directory exists if path includes folders
        os.makedirs(output_directory, exist_ok=True)
        
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Preprocessed data saved to: '{output_csv_path}'")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")
        
    return df 
