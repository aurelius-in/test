import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations

# Define the file paths (update these paths as per your directory structure)
feature_dir = '/path/to/your/feature_directory/'  # Update with your directory
binary_dir = '/path/to/your/binary_directory/'  # Update with your directory

# Load feature rankings and data
feature_rank = pd.read_csv(feature_dir + 'feature_rank.csv')
data = pd.read_csv(binary_dir + 'binary_500_case_status.csv')

# Define target variable
target_variable = 'Risk'

# Preprocess the data
data.fillna(method='ffill', inplace=True)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Splitting the dataset into features and target variable
X = data.drop(target_variable, axis=1)
y = data[target_variable]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to train and evaluate SVM model
def evaluate_model(features):
    model = SVC(kernel='rbf', gamma='auto')
    model.fit(X_train[:, features], y_train)
    predictions = model.predict(X_test[:, features])
    return accuracy_score(y_test, predictions)

# Get all feature names
all_features = feature_rank['Feature'].tolist()

# Get indices of top 14 features
top_14_indices = [all_features.index(feature) for feature in feature_rank.sort_values(by='LLM SVM Rank').head(14)['Feature'].tolist()]

# Start with best combination of 10 out of top 14 features
best_accuracy = 0
best_features_indices = []
for combination in combinations(top_14_indices, 10):
    accuracy = evaluate_model(list(combination))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features_indices = list(combination)

print(f'Best initial combination: {best_features_indices} with accuracy: {best_accuracy}')

# Include the remaining features
for index in range(len(all_features)):
    if index not in best_features_indices:
        current_features_indices = best_features_indices + [index]
        accuracy = evaluate_model(current_features_indices)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features_indices = current_features_indices
            print(f'Adding feature at index {index} improved accuracy to {best_accuracy}')

print(f'Final set of features: {best_features_indices} with accuracy: {best_accuracy}')
