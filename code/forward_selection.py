import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
feature_rank_df = pd.read_csv(feature_dir + 'feature_rank.csv')
synthetic_data = pd.read_csv(output_dir + 'smote2000_96features.csv')
real_data = pd.read_csv(binary_dir + 'real500.csv')

# Define target variable
target_variable = 'Risk'

# Starting with top 14 features
top_features = []
for value in feature_rank_df['LLM SVM Rank'].iloc[:14].tolist():
    feature = feature_rank_df.loc[feature_rank_df['LLM SVM Rank'] == value, 'Feature'].item()
    top_features.append(feature)


# Train initial model
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_real[top_features])
initial_accuracy = accuracy_score(y_real, y_pred)
print(f'Initial model accuracy with top 14 features: {initial_accuracy}')

# Forward selection process
for feature in feature_rank_df['LLM SVM Rank'].iloc[14:96]:
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
# Train initial model
X_train = synthetic_data[top_features]
y_train = synthetic_data[target_variable]
X_test = real_data[top_features]
y_test = real_data[target_variable]

model = SVC()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
