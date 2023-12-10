import pandas as pd

# Load the raw data
raw_data = pd.read_csv('provider_actions_raw.csv')

# Load additional data
comments_data = pd.read_csv('features/features_comments.csv')
pass_data = pd.read_csv('features_pass.csv')
recent_case_data = pd.read_csv('features/features_recent_case.csv')

# Merge the raw data with comments scores
data_with_comments = pd.merge(raw_data, comments_data[['Provider ID', 'Comments Score']], on='Provider ID')

# Merge with pass scores
data_with_pass = pd.merge(data_with_comments, pass_data[['Provider ID', 'Reason for Pass Score']], on='Provider ID')

# Merge with recent case scores
final_data = pd.merge(data_with_pass, recent_case_data[['Provider ID', 'OpenDtScore', 'CloseDtScore', 'DataMiningUpdateDtScore']], on='Provider ID')

# Rename columns to match your final dataset
final_data.rename(columns={
    'OpenDtScore': 'Most Recent Case Open Dt',
    'CloseDtScore': 'Most Recent Case Close Dt',
    'DataMiningUpdateDtScore': 'Most Recent Data Mining Activity Update Dt'
}, inplace=True)

# Save the merged data
final_data.to_csv('provider_actions.csv', index=False)

print("Data processing complete. The merged data is saved in 'provider_actions.csv'.")
