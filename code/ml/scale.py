import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming labeled_data and unlabeled_data are your DataFrames
# Replace with your actual DataFrame names

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Function to check if scaling is needed
def needs_scaling(column):
    return column.min() < 0 or column.max() > 1

# Apply scaling to the DataFrames
for df in [labeled_data, unlabeled_data]:
    for col in df.columns:
        if col != 'PRCP ID' and needs_scaling(df[col]):
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
