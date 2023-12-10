import pandas as pd

# Load your CSV file
csv_file = 'labeled_providers.csv'  # CSV file path
data = pd.read_csv(csv_file)

# Define the weights for each feature
weights = {
    'Most Recent Case Status': 0.25,
    'Reason for Pass Score': 0.20,
    'Comments Score': 0.15,
    'Most Recent Closed Status Score': 0.10,

}

# Normalize function (example: Min-Max Scaling)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Calculate default values (mean, median, etc.) for missing data
default_values = {}
for feature in weights.keys():
    default_values[feature] = data[feature].mean()  # or median(), or a predefined value

# Function to calculate risk score
def calculate_risk_score(row):
    risk_score = 0
    for feature, weight in weights.items():
        feature_value = row.get(feature, default_values[feature])
        # Assuming all values are already normalized, otherwise call normalize function
        risk_score += feature_value * weight
    return risk_score * 100  # Normalize to a 0-100 scale

# Apply the function to each row in the dataset
data['Risk Score'] = data.apply(calculate_risk_score, axis=1)

# Output the result
print(data[['Provider ID', 'Risk Score']])  # Assuming there's a 'Provider ID' column
