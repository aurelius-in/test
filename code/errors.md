ValueError                                Traceback (most recent call last)
Cell In[43], line 40
     38 scaler = StandardScaler()
     39 X_synthetic_scaled = scaler.fit_transform(X_synthetic)
---> 40 X_real_scaled = scaler.transform(X_real)
     42 # Function to train and evaluate SVM model
     43 def evaluate_model(feature_indices):

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/sklearn/utils/_set_output.py:140, in _wrap_method_output.<locals>.wrapped(self, X, *args, **kwargs)
    138 @wraps(f)
    139 def wrapped(self, X, *args, **kwargs):
--> 140     data_to_wrap = f(self, X, *args, **kwargs)
    141     if isinstance(data_to_wrap, tuple):
    142         # only wrap the first output for cross decomposition
    143         return (
    144             _wrap_data_with_container(method, data_to_wrap[0], X, self),
    145             *data_to_wrap[1:],
    146         )

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/sklearn/preprocessing/_data.py:992, in StandardScaler.transform(self, X, copy)
    989 check_is_fitted(self)
    991 copy = copy if copy is not None else self.copy
--> 992 X = self._validate_data(
    993     X,
    994     reset=False,
    995     accept_sparse="csr",
    996     copy=copy,
    997     dtype=FLOAT_DTYPES,
    998     force_all_finite="allow-nan",
    999 )
   1001 if sparse.issparse(X):
   1002     if self.with_mean:

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/sklearn/base.py:548, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, **check_params)
    483 def _validate_data(
    484     self,
    485     X="no_validation",
   (...)
    489     **check_params,
    490 ):
    491     """Validate input data and set or check the `n_features_in_` attribute.
    492 
    493     Parameters
   (...)
    546         validated.
    547     """
--> 548     self._check_feature_names(X, reset=reset)
    550     if y is None and self._get_tags()["requires_y"]:
    551         raise ValueError(
    552             f"This {self.__class__.__name__} estimator "
    553             "requires y to be passed, but the target y is None."
    554         )

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/sklearn/base.py:481, in BaseEstimator._check_feature_names(self, X, reset)
    476 if not missing_names and not unexpected_names:
    477     message += (
    478         "Feature names must be in the same order as they were in fit.\n"
    479     )
--> 481 raise ValueError(message)

ValueError: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.
