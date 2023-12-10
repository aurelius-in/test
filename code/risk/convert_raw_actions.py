import pandas as pd

# File locations
provider_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/providers/"
feature_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/features/"
labels_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/labels/"

# Load the raw data
raw_data = pd.read_csv(provider_dir + 'provider_actions_raw.csv')

# Load additional data 
comments_data = pd.read_csv(feature_dir +'features_comments.csv')
pass_data = pd.read_csv(feature_dir + 'features_pass.csv')
recent_case_data = pd.read_csv(feature_dir + 'features_recent_case.csv')

# Merge the raw data with comments scores
data_with_comments = pd.merge(raw_data, comments_data[['Comments', 'Comment Score']], on='Comments')

# Merge with pass scores
data_with_pass = pd.merge(data_with_comments, pass_data[['Reason for pass', 'Pass Score']], on='Pass Score')

# Merge with recent case scores
final_data = pd.merge(data_with_pass, recent_case_data[['Year', 'OpenDtScore', 'CloseDtScore', 'DataMiningUpdateDtScore']], on='Year')

# Rename columns to match your final dataset
final_data.rename(columns={
    'OpenDtScore': 'Most Recent Case Open Dt',
    'CloseDtScore': 'Most Recent Case Close Dt',
    'DataMiningUpdateDtScore': 'Most Recent Data Mining Activity Update Dt'
}, inplace=True)

# Save the merged data
final_data.to_csv(labels_dir + 'provider_actions.csv', index=False)

print("Data processing complete. The merged data is saved in 'provider_actions.csv'.")
