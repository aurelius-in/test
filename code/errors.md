KeyError                                  Traceback (most recent call last)
Cell In[13], line 47
     45 final_data = data_with_pass.copy()
     46 for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
---> 47     temp_data = pd.merge(final_data, recent_case_data[['Year', date_col + 'Score']], left_on=date_col + ' Year', right_on='Year')
     48     final_data[date_col + ' Score'] = temp_data[date_col + ' Score']
     50 # Drop extra Year columns

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/frame.py:3767, in DataFrame.__getitem__(self, key)
   3765     if is_iterator(key):
   3766         key = list(key)
-> 3767     indexer = self.columns._get_indexer_strict(key, "columns")[1]
   3769 # take() does not accept boolean indexers
   3770 if getattr(indexer, "dtype", None) == bool:

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/indexes/base.py:5876, in Index._get_indexer_strict(self, key, axis_name)
   5873 else:
   5874     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
-> 5876 self._raise_if_missing(keyarr, indexer, axis_name)
   5878 keyarr = self.take(indexer)
   5879 if isinstance(key, Index):
   5880     # GH 42790 - Preserve name from an Index

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/indexes/base.py:5938, in Index._raise_if_missing(self, key, indexer, axis_name)
   5935     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   5937 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 5938 raise KeyError(f"{not_found} not in index")

KeyError: "['Most Recent Case Open DtScore'] not in index"
