import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
feature_rank_df = pd.read_csv('feature_rank.csv')
synthetic_data = pd.read_csv('smote2000.csv')
real_data = pd.read_csv('real500.csv')

# Define target variable name
target_variable = 'target'  # Replace with the name of your target variable

# Splitting real data into features and target
X_real = real_data.drop(target_variable, axis=1)
y_real = real_data[target_variable]

# Starting with top 14 features
top_features = feature_rank_df['LLM CSV Rank'].iloc[:14].tolist()
X_train = synthetic_data[top_features]
y_train = synthetic_data[target_variable]

# Train initial model
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_real[top_features])
initial_accuracy = accuracy_score(y_real, y_pred)
print(f'Initial model accuracy with top 14 features: {initial_accuracy}')

# Forward selection process
for feature in feature_rank_df['LLM CSV Rank'].iloc[14:96]:
    new_features = top_features + [feature]
    X_train_new = synthetic_data[new_features]
    
    # Train model with the new set of features
    model.fit(X_train_new, y_train)
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
