import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load data
feature_rank_df = pd.read_csv(feature_dir + 'feature_rank.csv')
synthetic_data = pd.read_csv(output_dir + 'smote2000_96features.csv')
real_data = pd.read_csv(binary_dir + 'real500.csv')

# Exclude 'PRCP ID' column if it exists
exclude_column = 'PRCP ID'
if exclude_column in synthetic_data.columns:
    synthetic_data.drop(columns=[exclude_column], inplace=True)
if exclude_column in real_data.columns:
    real_data.drop(columns=[exclude_column], inplace=True)

# Preprocess the data
label_encoders = {}
for column in synthetic_data.columns[:-1]:  # Loop through all but the last column
    if synthetic_data[column].dtype.name == 'object':  # Corrected line
        le = LabelEncoder()
        synthetic_data[column] = le.fit_transform(synthetic_data[column])
        real_data[column] = le.transform(real_data[column])
        label_encoders[column] = le
        
# Split datasets into features and target variable
X_synthetic = synthetic_data.drop('Risk', axis=1)
y_synthetic = synthetic_data['Risk']
X_real = real_data.drop('Risk', axis=1)
y_real = real_data['Risk']

# Standardize the features
scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)

# Function to train and evaluate SVM model
def evaluate_model(feature_indices):
    model = SVC(kernel='rbf', gamma='auto')
    model.fit(X_synthetic_scaled[:, feature_indices], y_synthetic)
    predictions = model.predict(X_real_scaled[:, feature_indices])
    return accuracy_score(y_real, predictions)

# Get all feature indices excluding 'PRCP ID'
all_feature_indices = list(range(X_synthetic.shape[1]))

# Start with 1 feature and add more
best_accuracy = 0
best_features_indices = []
accuracy_progress = []

for combination in combinations(all_feature_indices, 1):
    current_comb = list(combination)
    accuracy = evaluate_model(current_comb)
    print(f'Trying features {current_comb}, Accuracy: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features_indices = current_comb
        print(f'New best feature set {best_features_indices} with accuracy {best_accuracy}')
    
    # Iteratively add more features
    for feature_index in all_feature_indices:
        if feature_index not in best_features_indices:
            new_comb = best_features_indices + [feature_index]
            new_accuracy = evaluate_model(new_comb)
            print(f'Trying features {new_comb}, Accuracy: {new_accuracy}')
            
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_features_indices = new_comb
                print(f'New best feature set {best_features_indices} with accuracy {best_accuracy}')
    
    accuracy_progress.append(best_accuracy)

# Plot accuracy progression
plt.figure(figsize=(10, 6))
plt.plot(accuracy_progress, marker='o')
plt.title('Model Accuracy Progression')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

print(f'Final best feature set: {best_features_indices} with accuracy {best_accuracy}')
