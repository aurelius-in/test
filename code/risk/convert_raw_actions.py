import pandas as pd
from datetime import datetime, timedelta

# Function to convert Excel-style dates, or regular dates
def convert_excel_date(excel_date):
    try:
        return pd.to_datetime(excel_date, errors='coerce', format='%m/%d/%Y')
    except ValueError:
        # Assuming Excel date format (number of days since 1900-01-01)
        try:
            return datetime(1899, 12, 30) + timedelta(days=float(excel_date))
        except ValueError:
            # Return None if the conversion is not possible
            return None

# File locations
provider_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/providers/"
feature_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/features/"
labels_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/pod-mlcomputeinstance2/code/Data/labels/"

# Load the raw data
raw_data = pd.read_csv(provider_dir + 'provider_actions_raw.csv')

# Convert and extract years from date columns in raw data
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    raw_data[date_col] = raw_data[date_col].apply(convert_excel_date)
    raw_data[date_col + ' Year'] = raw_data[date_col].dt.year

# Load additional data 
comments_data = pd.read_csv(feature_dir +'features_comments.csv')
pass_data = pd.read_csv(feature_dir + 'features_pass.csv')
recent_case_data = pd.read_csv(feature_dir + 'features_recent_case.csv')
status_data = pd.read_csv(feature_dir + 'features_status.csv')

# Merge the raw data with Case Status scores
data_with_status = pd.merge(raw_data, status_data[['Most Recent Case Status', 'Status Score']], on='Most Recent Case Status')

# Merge the raw data with comments scores
data_with_comments = pd.merge(raw_data, comments_data[['Comments', 'Comment Score']], on='Comments')

# Merge with pass scores
data_with_pass = pd.merge(data_with_comments, pass_data[['Reason for pass', 'Pass Score']], on='Reason for pass')

# Merge with recent case scores
final_data = data_with_pass.copy()
for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
    temp_data = pd.merge(final_data, recent_case_data[['Year', date_col + 'Score']], left_on=date_col + ' Year', right_on='Year')
    final_data[date_col + ' Score'] = temp_data[date_col + ' Score']

# Drop extra Year columns
final_data.drop(columns=['Most Recent Case Open Dt Year', 'Most Recent Case Close Dt Year', 'Most Recent Data Mining Activity Update Dt Year'], inplace=True)

# Rename columns to match your final dataset
final_data.rename(columns={
    'Most Recent Case Status Score': 'StatusScore',
    'Pass Score': 'PassScore',
    'Comments Score': 'CommentsScore',
    'Most Recent Case Open Dt Score': 'OpenDtScore',
    'Most Recent Case Close Dt Score': 'CloseDtScore',
    'Most Recent Data Mining Activity Update Dt Score': 'MiningDtScore'
}, inplace=True)

# Save the merged data
final_data.to_csv(labels_dir + 'provider_actions.csv', index=False)

print("Data processing complete. The merged data is saved in 'provider_actions.csv'.")
