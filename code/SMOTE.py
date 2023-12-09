from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load your dataset
data = pd.read_csv('path_to_your_data/real500.csv')

# Assuming 'Risk' is your target variable
X = data.drop('Risk', axis=1)
y = data['Risk']

# Split your data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(sampling_strategy={0: 1000, 1: 1000})

# Apply SMOTE
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Combine the features and labels into one DataFrame
balanced_data = pd.concat([pd.DataFrame(X_smote, columns=X_train.columns), pd.DataFrame(y_smote, columns=['Risk'])], axis=1)

# Save the balanced data to a new CSV file
balanced_data.to_csv('path_to_your_data/synthetic_balanced_data.csv', index=False)
