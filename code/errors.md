ValueError                                Traceback (most recent call last)
Cell In[46], line 18
     16 # Generate synthetic data
     17 num_samples = 2000
---> 18 synthetic_data = np.random.randn(num_samples, X_real.shape[1]) * stds + means
     19 synthetic_data = pd.DataFrame(synthetic_data, columns=X_real.columns)
     21 # Generate synthetic labels, balancing the classes as the real dataset

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/generic.py:2016, in NDFrame.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   2012 @final
   2013 def __array_ufunc__(
   2014     self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
   2015 ):
-> 2016     return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/arraylike.py:273, in array_ufunc(self, ufunc, method, *inputs, **kwargs)
    270 kwargs = _standardize_out_kwarg(**kwargs)
    272 # for binary ops, use our custom dunder methods
--> 273 result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    274 if result is not NotImplemented:
    275     return result

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/_libs/ops_dispatch.pyx:113, in pandas._libs.ops_dispatch.maybe_dispatch_ufunc_to_dunder_op()

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/ops/common.py:81, in _unpack_zerodim_and_defer.<locals>.new_method(self, other)
     77             return NotImplemented
     79 other = item_from_zerodim(other)
---> 81 return method(self, other)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/arraylike.py:206, in OpsMixin.__rmul__(self, other)
    204 @unpack_zerodim_and_defer("__rmul__")
    205 def __rmul__(self, other):
--> 206     return self._arith_method(other, roperator.rmul)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/series.py:6112, in Series._arith_method(self, other, op)
   6110 def _arith_method(self, other, op):
   6111     self, other = ops.align_method_SERIES(self, other)
-> 6112     return base.IndexOpsMixin._arith_method(self, other, op)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/base.py:1350, in IndexOpsMixin._arith_method(self, other, op)
   1347 with np.errstate(all="ignore"):
   1348     result = ops.arithmetic_op(lvalues, rvalues, op)
-> 1350 return self._construct_result(result, name=res_name)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/series.py:3105, in Series._construct_result(self, result, name)
   3102 # TODO: result should always be ArrayLike, but this fails for some
   3103 #  JSONArray tests
   3104 dtype = getattr(result, "dtype", None)
-> 3105 out = self._constructor(result, index=self.index, dtype=dtype)
   3106 out = out.__finalize__(self)
   3108 # Set the result's name after __finalize__ is called because __finalize__
   3109 #  would set it back to self.name

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/series.py:500, in Series.__init__(self, data, index, dtype, name, copy, fastpath)
    498     index = default_index(len(data))
    499 elif is_list_like(data):
--> 500     com.require_length_match(data, index)
    502 # create/copy the manager
    503 if isinstance(data, (SingleBlockManager, SingleArrayManager)):

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/common.py:576, in require_length_match(data, index)
    572 """
    573 Check the length of data matches the length of the index.
    574 """
    575 if len(data) != len(index):
--> 576     raise ValueError(
    577         "Length of values "
    578         f"({len(data)}) "
    579         "does not match length of index "
    580         f"({len(index)})"
    581     )

ValueError: Length of values (2000) does not match length of index (97)
