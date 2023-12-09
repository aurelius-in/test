import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
import scikitplot as skplt

binary_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/binary/"
output_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/binary_output/"

# Load feature rankings
feature_rank = pd.read_csv(feature_dir + 'feature_rank.csv')

# Select top 14 features
top_features = feature_rank.sort_values(by='LLM SVM Rank').head(14)['Feature'].tolist()

# Load the training data
data = pd.read_csv(binary_dir + 'binary_500_case_status.csv')

# Filter for top 14 features, include 'Risk' column as well
data = data[top_features + ['Risk']]

# Preprocess the data
data.fillna(method='ffill', inplace=True)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Splitting the dataset into features and target variable
X = data.drop('Risk', axis=1)
y = data['Risk']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC(kernel='rbf', gamma='auto', probability=True)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("F1 Score:", f1_score(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("SVM Confusion Matrix: 14 LLM Features\n", conf_matrix)
skplt.metrics.plot_confusion_matrix(y_test, predictions)
plt.show()

# ROC Curve
probabilities = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, probabilities[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('SVM False Positive Rate')
plt.ylabel('SVM True Positive Rate')
plt.title('SVM Receiver Operating Characteristic: 14 LLM Featutes')
plt.legend(loc="lower right")
plt.show()

# Predict for unlabeled data
unlabeled_data = pd.read_csv(binary_dir + 'binary_unlabeled_providers.csv')

# Filter the unlabeled data to include only top 14 features
unlabeled_data = unlabeled_data[top_features]

# Preprocess the unlabeled data
unlabeled_data.fillna(method='ffill', inplace=True)
for column in unlabeled_data.columns:
    if unlabeled_data[column].dtype == 'object' and column in label_encoders:
        unlabeled_data[column] = label_encoders[column].transform(unlabeled_data[column])

# Standardize the features of unlabeled data
X_unlabeled = scaler.transform(unlabeled_data)

# Predict and add the 'Risk' column
unlabeled_data['Risk'] = model.predict(X_unlabeled)

# Save the updated dataframe
unlabeled_data.to_csv(output_dir + 'binary_labeled_providers14_LLM.csv', index=False)
