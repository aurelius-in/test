import pandas as pd

# Load the DataFrame
df = pd.read_csv('all_labeled_combined_with_average.csv')

# Ensure all data is numeric for correlation calculation
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract correlations with 'Label Score' and 'Feature Score'
label_score_correlation = correlation_matrix['Label Score']
feature_score_correlation = correlation_matrix['Feature Score']

# Get the top 36 features most positively correlated with 'Label Score'
top_36_label_score = label_score_correlation.nlargest(37).iloc[1:]

# Get the top 36 features most positively correlated with 'Feature Score'
top_36_feature_score = feature_score_correlation.nlargest(37).iloc[1:]

print("Top 36 features correlated with Label Score:")
print(top_36_label_score)
print("\nTop 36 features correlated with Feature Score:")
print(top_36_feature_score)
