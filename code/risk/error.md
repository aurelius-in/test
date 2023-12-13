/tmp/ipykernel_5864/250403437.py:4: DtypeWarning: Columns (44,101,104,105) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(labels_dir + 'all_labeled_combined.csv')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[3], line 13
     11 for col in df.columns:
     12     if col not in exclude_columns and col != 'PRCP ID':  # Excluding specified columns and 'PRCP ID'
---> 13         percentile_25th = df[col].quantile(0.25)
     14         df[col].fillna(percentile_25th, inplace=True)
     16 # Calculate the average for each row, excluding specified columns and 'PRCP ID'

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/series.py:2650, in Series.quantile(self, q, interpolation)
   2646 # We dispatch to DataFrame so that core.internals only has to worry
   2647 #  about 2D cases.
   2648 df = self.to_frame()
-> 2650 result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
   2651 if result.ndim == 2:
   2652     result = result.iloc[:, 0]

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/frame.py:10882, in DataFrame.quantile(self, q, axis, numeric_only, interpolation, method)
  10875 axis = self._get_axis_number(axis)
  10877 if not is_list_like(q):
  10878     # BlockManager.quantile expects listlike, so we wrap and unwrap here
  10879     # error: List item 0 has incompatible type "Union[float, Union[Union[
  10880     # ExtensionArray, ndarray[Any, Any]], Index, Series], Sequence[float]]";
  10881     # expected "float"
> 10882     res_df = self.quantile(  # type: ignore[call-overload]
  10883         [q],
  10884         axis=axis,
  10885         numeric_only=numeric_only,
  10886         interpolation=interpolation,
  10887         method=method,
  10888     )
  10889     if method == "single":
  10890         res = res_df.iloc[0]

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/frame.py:10927, in DataFrame.quantile(self, q, axis, numeric_only, interpolation, method)
  10923     raise ValueError(
  10924         f"Invalid method: {method}. Method must be in {valid_method}."
  10925     )
  10926 if method == "single":
> 10927     res = data._mgr.quantile(qs=q, axis=1, interpolation=interpolation)
  10928 elif method == "table":
  10929     valid_interpolation = {"nearest", "lower", "higher"}

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/internals/managers.py:1587, in BlockManager.quantile(self, qs, axis, interpolation)
   1584 new_axes = list(self.axes)
   1585 new_axes[1] = Index(qs, dtype=np.float64)
-> 1587 blocks = [
   1588     blk.quantile(axis=axis, qs=qs, interpolation=interpolation)
   1589     for blk in self.blocks
   1590 ]
   1592 return type(self)(blocks, new_axes)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/internals/managers.py:1588, in <listcomp>(.0)
   1584 new_axes = list(self.axes)
   1585 new_axes[1] = Index(qs, dtype=np.float64)
   1587 blocks = [
-> 1588     blk.quantile(axis=axis, qs=qs, interpolation=interpolation)
   1589     for blk in self.blocks
   1590 ]
   1592 return type(self)(blocks, new_axes)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/internals/blocks.py:1463, in Block.quantile(self, qs, interpolation, axis)
   1460 assert axis == 1  # only ever called this way
   1461 assert is_list_like(qs)  # caller is responsible for this
-> 1463 result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
   1464 # ensure_block_shape needed for cases where we start with EA and result
   1465 #  is ndarray, e.g. IntegerArray, SparseArray
   1466 result = ensure_block_shape(result, ndim=2)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/array_algos/quantile.py:37, in quantile_compat(values, qs, interpolation)
     35     fill_value = na_value_for_dtype(values.dtype, compat=False)
     36     mask = isna(values)
---> 37     return quantile_with_mask(values, mask, fill_value, qs, interpolation)
     38 else:
     39     return values._quantile(qs, interpolation)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/array_algos/quantile.py:95, in quantile_with_mask(values, mask, fill_value, qs, interpolation)
     93     result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
     94 else:
---> 95     result = _nanpercentile(
     96         values,
     97         qs * 100.0,
     98         na_value=fill_value,
     99         mask=mask,
    100         interpolation=interpolation,
    101     )
    103     result = np.array(result, copy=False)
    104     result = result.T

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/array_algos/quantile.py:196, in _nanpercentile(values, qs, na_value, mask, interpolation)
    193 if mask.any():
    194     # Caller is responsible for ensuring mask shape match
    195     assert mask.shape == values.shape
--> 196     result = [
    197         _nanpercentile_1d(val, m, qs, na_value, interpolation=interpolation)
    198         for (val, m) in zip(list(values), list(mask))
    199     ]
    200     if values.dtype.kind == "f":
    201         # preserve itemsize
    202         result = np.array(result, dtype=values.dtype, copy=False).T

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/array_algos/quantile.py:197, in <listcomp>(.0)
    193 if mask.any():
    194     # Caller is responsible for ensuring mask shape match
    195     assert mask.shape == values.shape
    196     result = [
--> 197         _nanpercentile_1d(val, m, qs, na_value, interpolation=interpolation)
    198         for (val, m) in zip(list(values), list(mask))
    199     ]
    200     if values.dtype.kind == "f":
    201         # preserve itemsize
    202         result = np.array(result, dtype=values.dtype, copy=False).T

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/array_algos/quantile.py:143, in _nanpercentile_1d(values, mask, qs, na_value, interpolation)
    137 if len(values) == 0:
    138     # Can't pass dtype=values.dtype here bc we might have na_value=np.nan
    139     #  with values.dtype=int64 see test_quantile_empty
    140     # equiv: 'np.array([na_value] * len(qs))' but much faster
    141     return np.full(len(qs), na_value)
--> 143 return np.percentile(
    144     values,
    145     qs,
    146     # error: No overload variant of "percentile" matches argument
    147     # types "ndarray[Any, Any]", "ndarray[Any, dtype[floating[_64Bit]]]"
    148     # , "Dict[str, str]"  [call-overload]
    149     **{np_percentile_argname: interpolation},  # type: ignore[call-overload]
    150 )

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/numpy/lib/function_base.py:4283, in percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation)
   4281 if not _quantile_is_valid(q):
   4282     raise ValueError("Percentiles must be in the range [0, 100]")
-> 4283 return _quantile_unchecked(
   4284     a, q, axis, out, overwrite_input, method, keepdims)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/numpy/lib/function_base.py:4555, in _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)
   4547 def _quantile_unchecked(a,
   4548                         q,
   4549                         axis=None,
   (...)
   4552                         method="linear",
   4553                         keepdims=False):
   4554     """Assumes that q is in [0, 1], and is an ndarray"""
-> 4555     return _ureduce(a,
   4556                     func=_quantile_ureduce_func,
   4557                     q=q,
   4558                     keepdims=keepdims,
   4559                     axis=axis,
   4560                     out=out,
   4561                     overwrite_input=overwrite_input,
   4562                     method=method)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/numpy/lib/function_base.py:3823, in _ureduce(a, func, keepdims, **kwargs)
   3820             index_out = (0, ) * nd
   3821             kwargs['out'] = out[(Ellipsis, ) + index_out]
-> 3823 r = func(a, **kwargs)
   3825 if out is not None:
   3826     return out

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/numpy/lib/function_base.py:4721, in _quantile_ureduce_func(a, q, axis, out, overwrite_input, method)
   4719     else:
   4720         arr = a.copy()
-> 4721 result = _quantile(arr,
   4722                    quantiles=q,
   4723                    axis=axis,
   4724                    method=method,
   4725                    out=out)
   4726 return result

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/numpy/lib/function_base.py:4823, in _quantile(arr, quantiles, axis, method, out)
   4819 previous_indexes, next_indexes = _get_indexes(arr,
   4820                                               virtual_indexes,
   4821                                               values_count)
   4822 # --- Sorting
-> 4823 arr.partition(
   4824     np.unique(np.concatenate(([0, -1],
   4825                               previous_indexes.ravel(),
   4826                               next_indexes.ravel(),
   4827                               ))),
   4828     axis=0)
   4829 if supports_nans:
   4830     slices_having_nans = np.isnan(arr[-1, ...])

