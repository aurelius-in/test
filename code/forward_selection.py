import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define directory paths (replace these with your actual directory paths)
feature_dir = 'path/to/feature_directory/'
output_dir = 'path/to/output_directory/'
binary_dir = 'path/to/binary_directory/'

# Load data
feature_rank_df = pd.read_csv(feature_dir + 'feature_rank.csv')
synthetic_data = pd.read_csv(output_dir + 'smote2000_96features.csv')
real_data = pd.read_csv(binary_dir + 'real500.csv')

# Define target variable
target_variable = 'Risk'

# Splitting real and synthetic data into features and target
X_real = real_data.drop(target_variable, axis=1)
y_real = real_data[target_variable]
X_synthetic = synthetic_data.drop(target_variable, axis=1)
y_synthetic = synthetic_data[target_variable]

# Starting with top 14 features
top_features = []
for value in feature_rank_df['LLM SVM Rank'].iloc[:14].tolist():
    feature = feature_rank_df.loc[feature_rank_df['LLM SVM Rank'] == value, 'Feature'].item()
    top_features.append(feature)

# Train initial model
model = SVC()
model.fit(X_synthetic[top_features], y_synthetic)
y_pred = model.predict(X_real[top_features])
initial_accuracy = accuracy_score(y_real, y_pred)
print(f'Initial model accuracy with top 14 features: {initial_accuracy}')

# Forward selection process
for rank in feature_rank_df['LLM SVM Rank'].iloc[14:96]:
    feature = feature_rank_df.loc[feature_rank_df['LLM SVM Rank'] == rank, 'Feature'].item()
    new_features = top_features + [feature]
    
    # Train model with the new set of features
    model.fit(X_synthetic[new_features], y_synthetic)
    y_pred_new = model.predict(X_real[new_features])
    new_accuracy = accuracy_score(y_real, y_pred_new)
    
    # Check if the new feature improves accuracy
    if new_accuracy > initial_accuracy:
        print(f'Adding feature {feature} improved accuracy to {new_accuracy}')
        top_features.append(feature)
        initial_accuracy = new_accuracy
    else:
        print(f'Feature {feature} did not improve accuracy, skipping...')

print(f'Final set of features: {top_features}')
