import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
synthetic_data = pd.read_csv('path_to_synthetic_data.csv')
real_data = pd.read_csv('path_to_real_data.csv')

# Drop 'PRCP ID' column if it exists in both datasets
exclude_column = 'PRCP ID'
synthetic_data.drop(columns=[exclude_column], inplace=True, errors='ignore')
real_data.drop(columns=[exclude_column], inplace=True, errors='ignore')

# Ensure both datasets have the same columns in the same order
common_columns = synthetic_data.columns.intersection(real_data.columns)
X_synthetic = synthetic_data[common_columns]
X_real = real_data[common_columns]

# Add target column
y_synthetic = synthetic_data['Risk']
y_real = real_data['Risk']

# Standardize the features
scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)
