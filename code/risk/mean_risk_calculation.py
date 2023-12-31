import pandas as pd
from datetime import datetime

# Load your CSV file
csv_file = 'provider_actions.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Define the weights for each feature, including date columns
weights = {
    'Most Recent Case Status': 0.50,
    'Reason for Pass Score': 0.25,
    'Comments Score': 0.20,
    'Most Recent Case Open Dt': 0.10,
    'Most Recent Case Close Dt': 0.10,
    'Most Recent Data Mining Activity Update Dt': 0.05
}

# Normalize function (example: Min-Max Scaling)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

# Convert date columns to numerical values (days since the date)
def convert_date_to_numerical(date_str):
    if pd.isna(date_str):
        return None
    date_format = "%Y-%m-%d"  # Adjust this based on your date format
    date_obj = datetime.strptime(date_str, date_format)
    return (datetime.now() - date_obj).days

# Convert and normalize date columns
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    data[date_col] = data[date_col].apply(convert_date_to_numerical)
    data[date_col] = data[date_col].fillna(data[date_col].mean())  # Replace NaN with mean
    min_val = data[date_col].min()
    max_val = data[date_col].max()
    data[date_col] = data[date_col].apply(lambda x: normalize(x, min_val, max_val))

# Calculate default values for missing data in other columns
default_values = {}
for feature in weights.keys():
    if feature not in data.columns:  # Skip if the feature is not in the dataset
        continue
    default_values[feature] = data[feature].mean()  # or median(), or a predefined value

# Function to calculate risk score
def calculate_risk_score(row):
    risk_score = 0
    for feature, weight in weights.items():
        feature_value = row.get(feature, default_values.get(feature, 0))
        risk_score += feature_value * weight
    return risk_score * 100  # Normalize to a 0-100 scale

# Apply the function to each row in the dataset
data['Risk Score'] = data.apply(calculate_risk_score, axis=1)

# Specify the file name for the new CSV
output_file = 'provider_actions_with_risk_scores.csv'

# Save the DataFrame to a new CSV, keeping the original columns plus the new 'Risk Score'
data.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")
