ValueError                                Traceback (most recent call last)
Cell In[24], line 22
     20 top_features = []
     21 for value in feature_rank_df['LLM SVM Rank'].iloc[:14].tolist():
---> 22     feature = feature_rank_df.loc[feature_rank_df['LLM SVM Rank'] == value, 'Feature'].item()
     23     top_features.append(feature)
     25 # Train initial model

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/base.py:350, in IndexOpsMixin.item(self)
    348 if len(self) == 1:
    349     return next(iter(self))
--> 350 raise ValueError("can only convert an array of size 1 to a Python scalar")

ValueError: can only convert an array of size 1 to a Python scalar
