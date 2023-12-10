import pandas as pd

# File locations
provider_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/providers/"
feature_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/features/"
labels_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/labels/"

# Load the raw data
raw_data = pd.read_csv(provider_dir + 'provider_actions_raw.csv')

# Extract years from date columns in raw data
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    raw_data[date_col + ' Year'] = pd.to_datetime(raw_data[date_col], errors='coerce', format='%m/%d/%Y').dt.year

# Load additional data 
status_data = pd.read_csv(feature_dir + 'features_status.csv')
comments_data = pd.read_csv(feature_dir + 'features_comments.csv')
pass_data = pd.read_csv(feature_dir + 'features_pass.csv')
recent_case_data = pd.read_csv(feature_dir + 'features_recent_case.csv')

# Merge the raw data with comments scores
data_with_comments = pd.merge(raw_data, comments_data[['Comments', 'Comment Score']], on='Comments')

# Merge with pass scores
data_with_pass = pd.merge(data_with_comments, pass_data[['Reason for pass', 'Pass Score']], on='Reason for pass')

# Merge the raw data with case status scores
data_with_status = pd.merge(data_with_pass, status_data[['Most Recent Case Status', 'Most Recent Case Status Score']], on='Most Recent Case Status')

# Merge with recent case scores
final_data = data_with_status.copy()
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    score_col_name = date_col + ' Score' 
    temp_data = pd.merge(final_data, recent_case_data[['Year', score_col_name]], left_on=date_col + ' Year', right_on='Year', how='left')
    final_data[score_col_name] = temp_data[score_col_name]

# Drop extra Year columns
final_data.drop(columns=['Reason for pass', 'Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt', 'Comments', 'Most Recent Case Status', 'Most Recent Case Open Dt Year', 'Most Recent Case Close Dt Year', 'Most Recent Data Mining Activity Update Dt Year'], inplace=True)

# Rename columns to match your final dataset
final_data.rename(columns={
    'Most Recent Case Status Score': 'StatusScore',
    'Comment Score': 'CommentsScore',
    'Pass Score': 'PassScore',
    'Most Recent Case Open Dt Score': 'OpenDtScore',
    'Most Recent Case Close Dt Score': 'CloseDtScore',
    'Most Recent Data Mining Activity Update Dt Score': 'MiningDtScore'
}, inplace=True)

# Save the merged data
final_data.to_csv(labels_dir + 'provider_actions.csv', index=False)

print("Data processing complete. The merged data is saved in 'provider_actions.csv'.")
