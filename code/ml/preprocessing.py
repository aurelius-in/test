
import pandas as pd

# Assuming raw_labeled and raw_unlabeled are defined in a previous cell block
# Load the datasets
labeled_data = pd.read_csv(raw_labeled)
unlabeled_data = pd.read_csv(raw_unlabeled)

# Remove the 'Provider Name' column
labeled_data = labeled_data.drop(columns=['Provider Name'])
unlabeled_data = unlabeled_data.drop(columns=['Provider Name'])

# Load the provider type mapping
provider_type_mapping = pd.read_csv('path/to/feature_provider_type.csv')

# Perform the merge
labeled_data_merged = labeled_data.merge(provider_type_mapping, on='Provider Type', how='left')
unlabeled_data_merged = unlabeled_data.merge(provider_type_mapping, on='Provider Type', how='left')

# Check the result of the merge
print(labeled_data_merged.head())
print(unlabeled_data_merged.head())

# Verify 'Weight' column exists after merge
if 'Weight' in labeled_data_merged and 'Weight' in unlabeled_data_merged:
    # Replace the 'Provider Type' column with the 'Weight' column
    labeled_data['Provider Type'] = labeled_data_merged['Weight']
    unlabeled_data['Provider Type'] = unlabeled_data_merged['Weight']
else:
    print("Weight column not found after merge. Check the consistency of Provider Type values.")

# Drop the 'Reason' column if it's not needed (assuming it's in the mapping file)
labeled_data.drop(columns=['Reason'], inplace=True, errors='ignore')
unlabeled_data.drop(columns=['Reason'], inplace=True, errors='ignore')
