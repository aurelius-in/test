import pandas as pd

# Load the CSV files
all_labeled_df = pd.read_csv('all_labeled_half_processed.csv')
provider_type_features = pd.read_csv('features_provider_type.csv')
states_features = pd.read_csv('features_states.csv')
region_features = pd.read_csv('features_region.csv')
in_ntwk_features = pd.read_csv('features_in_ntwl.csv')

# Create dictionaries for quick lookup
provider_type_dict = dict(zip(provider_type_features['Provider Type'], provider_type_features['Value']))
states_dict = dict(zip(states_features['State'], states_features['Weight']))
region_dict = dict(zip(region_features['Region'], region_features['Weight']))
in_ntwk_dict = dict(zip(in_ntwk_features['In Ntwk'], in_ntwk_features['Weight']))

# Map the values in all_labeled_df using these dictionaries
all_labeled_df['Provider Type'] = all_labeled_df['Provider Type'].map(provider_type_dict)
all_labeled_df['State'] = all_labeled_df['State'].map(states_dict)
all_labeled_df['Region'] = all_labeled_df['Region'].map(region_dict)
all_labeled_df['In Ntwk'] = all_labeled_df['In Ntwk'].map(in_ntwk_dict)

# Save the updated DataFrame to a new CSV file
all_labeled_df.to_csv('all_labeled_fully_processed.csv', index=False)
