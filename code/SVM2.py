import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations
import matplotlib.pyplot as plt

# Define file paths
feature_dir = '/path/to/your/feature_directory/'  # Update with your directory
synthetic_dir = '/path/to/your/synthetic_data_directory/'  # Update with your directory
real_dir = '/path/to/your/real_data_directory/'  # Update with your directory

# Load feature rankings and data
feature_rank = pd.read_csv(feature_dir + 'feature_rank.csv')
synthetic_data = pd.read_csv(synthetic_dir + 'synthetic_data.csv')
real_data = pd.read_csv(real_dir + 'real_data.csv')

# Exclude 'PRCP ID' column if it exists
exclude_column = 'PRCP ID'
if exclude_column in synthetic_data.columns:
    synthetic_data.drop(columns=[exclude_column], inplace=True)
if exclude_column in real_data.columns:
    real_data.drop(columns=[exclude_column], inplace=True)

# Preprocess the data (include steps for preprocessing)

# Define the target variable
target_variable = 'Risk'  # Update with the actual name of your target variable

# Split datasets into features and target variable
X_synthetic = synthetic_data.drop(target_variable, axis=1)
y_synthetic = synthetic_data[target_variable]
X_real = real_data.drop(target_variable, axis=1)
y_real = real_data[target_variable]

# Standardize the features
scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)

# Function to train and evaluate SVM model
def evaluate_model(features):
    model = SVC(kernel='rbf', gamma='auto')
    model.fit(X_synthetic_scaled[:, features], y_synthetic)
    predictions = model.predict(X_real_scaled[:, features])
    return accuracy_score(y_real, predictions)

# Get all feature indices excluding 'PRCP ID'
all_feature_indices = list(range(X_synthetic.shape[1]))

# Start with 1 feature and add more
best_accuracy = 0
best_features = []
accuracy_progress = []

for combination in combinations(all_feature_indices, 1):
    current_comb = list(combination)
    accuracy = evaluate_model(current_comb)
    print(f'Trying features {current_comb}, Accuracy: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = current_comb
        print(f'New best feature set {best_features} with accuracy {best_accuracy}')
    
    # Iteratively add more features
    for feature_index in all_feature_indices:
        if feature_index not in best_features:
            new_comb = best_features + [feature_index]
            new_accuracy = evaluate_model(new_comb)
            print(f'Trying features {new_comb}, Accuracy: {new_accuracy}')
            
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_features = new_comb
                print(f'New best feature set {best_features} with accuracy {best_accuracy}')
    
    accuracy_progress.append(best_accuracy)

# Plot accuracy progression
plt.figure(figsize=(10, 6))
plt.plot(accuracy_progress, marker='o')
plt.title('Model Accuracy Progression')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

print(f'Final best feature set: {best_features} with accuracy {best_accuracy}')
