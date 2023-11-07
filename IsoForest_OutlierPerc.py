import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the data
data = pd.read_csv('providers_data.csv')

# Iterate through each column
for column in data.columns:
    # Check if the column contains non-integer values
    if data[column].dtype != 'int':
        # Remove non-integer characters from each cell in the column
        data[column] = data[column].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
        try:
            # Convert the cleaned column to integer data type
            data[column] = data[column].astype(int)
        except ValueError:
            # If conversion fails, drop the column
            data = data.drop(columns=[column])

# Train an Isolation Forest model
X = data.drop(columns=['Outlier_Percent'])  # Assuming 'Outlier_Percent' is the target column
model = IsolationForest(contamination=0.05)
model.fit(X)

# Predict outliers (-1) and inliers (1)
predictions = model.predict(X)

# Create the 'Outlier_Percent' column with the predictions
data['Outlier_Percent'] = predictions

# Reorder columns to make 'Outlier_Percent' the last column
data = data[list(data.columns[:-1]) + ['Outlier_Percent']]

# Save the updated DataFrame to 'Outlier_Percent.csv'
data.to_csv('Outlier_Percent.csv', index=False)
