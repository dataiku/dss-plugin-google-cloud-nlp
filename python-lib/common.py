# -*- coding: utf-8 -*-
import itertools
import logging
import inspect
import functools
import time
import math
import numpy as np
import pandas as pd

from typing import Callable, List, Dict, Tuple, Type, AnyStr, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

API_RATE_LIMIT_PERIOD = 60  # 1 minute
API_RATE_LIMIT_QUOTA = 600  # 600 calls per period

API_EXCEPTIONS = Exception
try:
    from requests.exceptions import RequestException
    API_EXCEPTIONS = RequestException
except ImportError:
    pass
try:
    # TODO check if I can use even more specific errors
    from boto3.exceptions import Boto3Error
    from botocore.exceptions import BotoCoreError
    API_EXCEPTIONS = (Boto3Error, BotoCoreError)
except ImportError:
    pass
try:
    from google.api_core.exceptions import GoogleAPICallError, RetryError
    API_EXCEPTIONS = (GoogleAPICallError, RetryError)
except ImportError:
    pass
try:
    from azure.cognitiveservices.vision.computervision.models._models_py3 import ComputerVisionError, ComputerVisionErrorException
    API_EXCEPTIONS = (ComputerVisionError, ComputerVisionErrorException)
except ImportError:
    pass


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
            break
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def fail_or_warn_on_row(error_handling="fail", api_exceptions=API_EXCEPTIONS):
    if error_handling not in ["warn", "fail"]:
        raise ValueError(
            "Error handling parameter can be either 'warn' or 'fail'.")

    def inner_decorator(func):
        if "row" not in inspect.getfullargspec(func).args:
            raise ValueError("Function must have 'row' as first parameter.")

        def wrapped(row, *args, **kwargs):
            if not (isinstance(row, dict) or (isinstance(row, list) and isinstance(row[0], dict))):
                raise TypeError(
                    "The 'row' parameter must be a dict or a list of dict.")
            response_key = generate_unique("raw_response", row.keys())
            error_key = generate_unique("error", row.keys())
            if error_handling == 'fail':
                row[response_key] = func(row=row, *args, **kwargs)
                return row
            else:
                row[response_key] = ''
                row[error_key] = ''
                try:
                    row[response_key] = func(row=row, *args, **kwargs)
                    return row
                except api_exceptions as e:
                    logging.warning(str(e))
                    row[error_key] = str(e)
                    return row

        return wrapped

    return inner_decorator


def api_parallelizer(input_df:           pd.DataFrame,
                     api_call_function:   Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
                     parallel_workers:    int = 5,
                     api_support_batch:   bool = False,
                     batch_size:          int = 10,
                     **api_call_function_kwargs
                     ) -> pd.DataFrame:
    df_iterator = (i[1].to_dict() for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    if api_support_batch:
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    response_column = generate_unique("raw_response", input_df.columns)
    error_column = generate_unique("error", input_df.columns)
    output_schema = {**{response_column: str, error_column: str},
                     **dict(input_df.dtypes)}
    results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        futures = [pool.submit(fn=api_call_function, row=row, **api_call_function_kwargs)
                   for row in df_iterator]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            results.append(f.result())
    if api_support_batch:
        results = flatten(results)
    record_list = [
        {col: result.get(col) for col in output_schema.keys()} for result in results]
    output_df = pd.DataFrame.from_records(record_list) \
        .astype(output_schema) \
        .reindex(columns=list(input_df.columns) + [response_column, error_column])
    return output_df
