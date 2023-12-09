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
synthetic_data = pd.read_csv(output_dir + 'synthetic2000.csv')
real_data = pd.read_csv(binary_dir + 'real500.csv')

# Drop 'PRCP ID' column if it exists
exclude_column = 'PRCP ID'
synthetic_data.drop(columns=[exclude_column], inplace=True, errors='ignore')
real_data.drop(columns=[exclude_column], inplace=True, errors='ignore')

# Ensure both datasets have the same columns in the same order
common_columns = synthetic_data.columns.intersection(real_data.columns)
X_synthetic = synthetic_data[common_columns]
X_real = real_data[common_columns]

# Ensure 'Risk' column is in both datasets
y_synthetic = synthetic_data['Risk']
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

Trying features [0, 41], Accuracy: 1.0
Trying features [0, 42], Accuracy: 1.0
Trying features [0, 43], Accuracy: 0.996
Trying features [0, 44], Accuracy: 1.0
Trying features [0, 45], Accuracy: 1.0
Trying features [0, 46], Accuracy: 0.998
Trying features [0, 47], Accuracy: 0.998
Trying features [0, 48], Accuracy: 1.0
Trying features [0, 49], Accuracy: 1.0
Trying features [0, 50], Accuracy: 1.0
Trying features [0, 51], Accuracy: 1.0
Trying features [0, 52], Accuracy: 1.0
Trying features [0, 53], Accuracy: 0.998
Trying features [0, 54], Accuracy: 0.998
Trying features [0, 55], Accuracy: 1.0
Trying features [0, 56], Accuracy: 0.996
Trying features [0, 57], Accuracy: 1.0
Trying features [0, 58], Accuracy: 1.0
Trying features [0, 59], Accuracy: 0.998
Trying features [0, 60], Accuracy: 0.998
Trying features [0, 61], Accuracy: 1.0
Trying features [0, 62], Accuracy: 1.0
Trying features [0, 63], Accuracy: 0.998
Trying features [0, 64], Accuracy: 1.0
Trying features [0, 65], Accuracy: 1.0
Trying features [0, 66], Accuracy: 1.0
Trying features [0, 67], Accuracy: 1.0
Trying features [0, 68], Accuracy: 1.0
Trying features [0, 69], Accuracy: 0.998
Trying features [0, 70], Accuracy: 0.998
Trying features [0, 71], Accuracy: 0.998
Trying features [0, 72], Accuracy: 1.0
Trying features [0, 73], Accuracy: 1.0
Trying features [0, 74], Accuracy: 1.0
Trying features [0, 75], Accuracy: 1.0
Trying features [0, 76], Accuracy: 0.998
Trying features [0, 77], Accuracy: 0.998
Trying features [0, 78], Accuracy: 0.998
Trying features [0, 79], Accuracy: 0.998
Trying features [0, 80], Accuracy: 1.0
Trying features [0, 81], Accuracy: 1.0
Trying features [0, 82], Accuracy: 1.0
Trying features [0, 83], Accuracy: 0.998
Trying features [0, 84], Accuracy: 1.0
Trying features [0, 85], Accuracy: 0.998
Trying features [0, 86], Accuracy: 1.0
Trying features [0, 87], Accuracy: 1.0
Trying features [0, 88], Accuracy: 1.0
Trying features [0, 89], Accuracy: 0.998
Trying features [0, 90], Accuracy: 0.998
Trying features [0, 91], Accuracy: 0.998
Trying features [0, 92], Accuracy: 0.998
Trying features [0, 93], Accuracy: 1.0
Trying features [0, 94], Accuracy: 1.0
Trying features [0, 95], Accuracy: 1.0
Trying features [0, 96], Accuracy: 0.998
Trying features [96], Accuracy: 0.504
Trying features [0, 1], Accuracy: 1.0
Trying features [0, 2], Accuracy: 1.0
Trying features [0, 3], Accuracy: 1.0
Trying features [0, 4], Accuracy: 1.0
Trying features [0, 5], Accuracy: 0.998
Trying features [0, 6], Accuracy: 1.0
Trying features [0, 7], Accuracy: 1.0
Trying features [0, 8], Accuracy: 1.0
Trying features [0, 9], Accuracy: 1.0
Trying features [0, 10], Accuracy: 1.0
Trying features [0, 11], Accuracy: 0.998
Trying features [0, 12], Accuracy: 0.998
Trying features [0, 13], Accuracy: 0.998
Trying features [0, 14], Accuracy: 1.0
Trying features [0, 15], Accuracy: 1.0
Trying features [0, 16], Accuracy: 0.998
Trying features [0, 17], Accuracy: 1.0
Trying features [0, 18], Accuracy: 1.0
Trying features [0, 19], Accuracy: 1.0
Trying features [0, 20], Accuracy: 0.998
Trying features [0, 21], Accuracy: 0.998
Trying features [0, 22], Accuracy: 1.0
Trying features [0, 23], Accuracy: 1.0
Trying features [0, 24], Accuracy: 1.0
Trying features [0, 25], Accuracy: 1.0
Trying features [0, 26], Accuracy: 1.0
Trying features [0, 27], Accuracy: 0.998
Trying features [0, 28], Accuracy: 1.0
Trying features [0, 29], Accuracy: 0.998
Trying features [0, 30], Accuracy: 1.0
Trying features [0, 31], Accuracy: 0.998
Trying features [0, 32], Accuracy: 1.0
Trying features [0, 33], Accuracy: 1.0
Trying features [0, 34], Accuracy: 1.0
Trying features [0, 35], Accuracy: 0.998
Trying features [0, 36], Accuracy: 1.0
Trying features [0, 37], Accuracy: 1.0
Trying features [0, 38], Accuracy: 1.0
Trying features [0, 39], Accuracy: 1.0
Trying features [0, 40], Accuracy: 0.998
Trying features [0, 41], Accuracy: 1.0
Trying features [0, 42], Accuracy: 1.0
Trying features [0, 43], Accuracy: 0.996
Trying features [0, 44], Accuracy: 1.0
Trying features [0, 45], Accuracy: 1.0
Trying features [0, 46], Accuracy: 0.998
Trying features [0, 47], Accuracy: 0.998
Trying features [0, 48], Accuracy: 1.0
Trying features [0, 49], Accuracy: 1.0
Trying features [0, 50], Accuracy: 1.0
Trying features [0, 51], Accuracy: 1.0
Trying features [0, 52], Accuracy: 1.0
Trying features [0, 53], Accuracy: 0.998
Trying features [0, 54], Accuracy: 0.998
Trying features [0, 55], Accuracy: 1.0
Trying features [0, 56], Accuracy: 0.996
Trying features [0, 57], Accuracy: 1.0
Trying features [0, 58], Accuracy: 1.0
Trying features [0, 59], Accuracy: 0.998
Trying features [0, 60], Accuracy: 0.998
Trying features [0, 61], Accuracy: 1.0
Trying features [0, 62], Accuracy: 1.0
Trying features [0, 63], Accuracy: 0.998
Trying features [0, 64], Accuracy: 1.0
Trying features [0, 65], Accuracy: 1.0
Trying features [0, 66], Accuracy: 1.0
Trying features [0, 67], Accuracy: 1.0
Trying features [0, 68], Accuracy: 1.0
Trying features [0, 69], Accuracy: 0.998
Trying features [0, 70], Accuracy: 0.998
Trying features [0, 71], Accuracy: 0.998
Trying features [0, 72], Accuracy: 1.0
Trying features [0, 73], Accuracy: 1.0
Trying features [0, 74], Accuracy: 1.0
Trying features [0, 75], Accuracy: 1.0
Trying features [0, 76], Accuracy: 0.998
Trying features [0, 77], Accuracy: 0.998
Trying features [0, 78], Accuracy: 0.998
Trying features [0, 79], Accuracy: 0.998
Trying features [0, 80], Accuracy: 1.0
Trying features [0, 81], Accuracy: 1.0
Trying features [0, 82], Accuracy: 1.0
Trying features [0, 83], Accuracy: 0.998
Trying features [0, 84], Accuracy: 1.0
Trying features [0, 85], Accuracy: 0.998
Trying features [0, 86], Accuracy: 1.0
Trying features [0, 87], Accuracy: 1.0
Trying features [0, 88], Accuracy: 1.0
Trying features [0, 89], Accuracy: 0.998
Trying features [0, 90], Accuracy: 0.998
Trying features [0, 91], Accuracy: 0.998
Trying features [0, 92], Accuracy: 0.998
Trying features [0, 93], Accuracy: 1.0
Trying features [0, 94], Accuracy: 1.0
Trying features [0, 95], Accuracy: 1.0
Trying features [0, 96], Accuracy: 0.998
