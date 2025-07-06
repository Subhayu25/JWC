File "/mount/src/hr-analytics/app.py", line 434, in <module>
    main()
    ~~~~^^
File "/mount/src/hr-analytics/app.py", line 417, in main
    df = universal_filters(load_data())
                           ~~~~~~~~~^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 219, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 261, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 320, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
File "/mount/src/hr-analytics/app.py", line 55, in load_data
    df = pd.read_csv(GITHUB_URL)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ~~~~~~~~~~^
        f,
        ^^
    ...<6 lines>...
        storage_options=self.options.get("storage_options", None),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/common.py", line 728, in get_handle
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
    ...<3 lines>...
        storage_options=storage_options,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/common.py", line 384, in _get_filepath_or_buffer
    with urlopen(req_info) as req:
         ~~~~~~~^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/io/common.py", line 289, in urlopen
    return urllib.request.urlopen(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.13/urllib/request.py", line 189, in urlopen
    return opener.open(url, data, timeout)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.13/urllib/request.py", line 495, in open
    response = meth(req, response)
File "/usr/local/lib/python3.13/urllib/request.py", line 604, in http_response
    response = self.parent.error(
        'http', request, response, code, msg, hdrs)
File "/usr/local/lib/python3.13/urllib/request.py", line 533, in error
    return self._call_chain(*args)
           ~~~~~~~~~~~~~~~~^^^^^^^
File "/usr/local/lib/python3.13/urllib/request.py", line 466, in _call_chain
    result = func(*args)
File "/usr/local/lib/python3.13/urllib/request.py", line 613, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
