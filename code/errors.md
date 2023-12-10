ValueError                                Traceback (most recent call last)
Cell In[5], line 18
     16 # Extract years from date columns in raw data
     17 for date_col in ['Most Recent Case Open Dt', 'Most Recent Case Close Dt', 'Most Recent Data Mining Activity Update Dt']:
---> 18     raw_data[date_col + ' Year'] = pd.to_datetime(raw_data[date_col]).dt.year
     20 # Merge the raw data with comments scores
     21 data_with_comments = pd.merge(raw_data, comments_data[['Comments', 'Comment Score']], on='Comments')

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/tools/datetimes.py:1050, in to_datetime(arg, errors, dayfirst, yearfirst, utc, format, exact, unit, infer_datetime_format, origin, cache)
   1048         result = arg.map(cache_array)
   1049     else:
-> 1050         values = convert_listlike(arg._values, format)
   1051         result = arg._constructor(values, index=arg.index, name=arg.name)
   1052 elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/tools/datetimes.py:453, in _convert_listlike_datetimes(arg, format, name, utc, unit, errors, dayfirst, yearfirst, exact)
    451 # `format` could be inferred, or user didn't ask for mixed-format parsing.
    452 if format is not None and format != "mixed":
--> 453     return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
    455 result, tz_parsed = objects_to_datetime64ns(
    456     arg,
    457     dayfirst=dayfirst,
   (...)
    461     allow_object=True,
    462 )
    464 if tz_parsed is not None:
    465     # We can take a shortcut since the datetime64 numpy array
    466     # is in UTC

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/core/tools/datetimes.py:484, in _array_strptime_with_fallback(arg, name, utc, fmt, exact, errors)
    473 def _array_strptime_with_fallback(
    474     arg,
    475     name,
   (...)
    479     errors: str,
    480 ) -> Index:
    481     """
    482     Call array_strptime, with fallback behavior depending on 'errors'.
    483     """
--> 484     result, timezones = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
    485     if any(tz is not None for tz in timezones):
    486         return _return_parsed_timezone_results(result, timezones, utc, name)

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/_libs/tslibs/strptime.pyx:530, in pandas._libs.tslibs.strptime.array_strptime()

File /anaconda/envs/azureml_py310_sdkv2/lib/python3.10/site-packages/pandas/_libs/tslibs/strptime.pyx:351, in pandas._libs.tslibs.strptime.array_strptime()

ValueError: time data "39778.72727" doesn't match format "%m/%d/%Y", at position 1621. You might want to try:
    - passing `format` if your strings have a consistent format;
    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;
    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this.
