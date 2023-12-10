import pandas as pd

# File locations
provider_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/providers/"
feature_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/features/"
labels_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/labels/"

# Load the raw data
raw_data = pd.read_csv(provider_dir + 'provider_actions_raw.csv')

# Load additional data 
comments_data = pd.read_csv(feature_dir + 'features_comments.csv')
pass_data = pd.read_csv(feature_dir + 'features_pass.csv')
recent_case_data = pd.read_csv(feature_dir + 'features_recent_case.csv')

# Extract years from date columns in raw data
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    raw_data[date_col + ' Year'] = pd.to_datetime(raw_data[date_col]).dt.year

# Merge the raw data with comments scores
data_with_comments = pd.merge(raw_data, comments_data[['Comments', 'Comment Score']], on='Comments')

# Merge with pass scores
data_with_pass = pd.merge(data_with_comments, pass_data[['Reason for pass', 'Pass Score']], on='Reason for pass')

# Merge with recent case scores
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    final_data = pd.merge(data_with_pass, recent_case_data[['Year', date_col + 'Score']], left_on=date_col + ' Year', right_on='Year')
    data_with_pass[date_col + ' Score'] = final_data[date_col + 'Score']

# Drop extra Year columns
data_with_pass.drop(columns=['Most Recent Case Open Dt Year', 'Most Recent Case Close Dt Year', 'Most Recent Data Mining Activity Update Dt Year'], inplace=True)

# Rename columns to match your final dataset
data_with_pass.rename(columns={
    'Most Recent Case Open Dt Score': 'Most Recent Case Open Dt',
    'Most Recent Case Close Dt Score': 'Most Recent Case Close Dt',
    'Most Recent Data Mining Activity Update Dt Score': 'Most Recent Data Mining Activity Update Dt'
}, inplace=True)

# Save the merged data
data_with_pass.to_csv(labels_dir + 'provider_actions.csv', index=False)

print("Data processing complete. The merged data is saved in 'provider_actions.csv'.")
