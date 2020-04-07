# -*- coding: utf-8 -*-
import logging
import inspect
import math
import json

from functools import wraps
from enum import Enum
from typing import Callable, AnyStr, List, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class ErrorHandlingEnum(Enum):
    FAIL = "fail"
    WARN = "warn"


API_EXCEPTIONS = Exception
try:
    from requests.exceptions import RequestException
    API_EXCEPTIONS = RequestException
except ImportError:
    pass
try:
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

# ==============================================================================
# FUNCTION DEFINITION
# ==============================================================================


def generate_unique(name: AnyStr, existing_names: List) -> AnyStr:
    """
    Generate a unique name among existing ones by appending a number.
    """
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
            break
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def safe_json_loads(
    str_to_check: AnyStr,
    error_handling: AnyStr = ErrorHandlingEnum.FAIL.value
) -> Dict:
    """
    Wrap json.loads with an additional parameter to handle errors:
    - 'fail' to use json.loads, which fails on invalid data
    - 'warn' to try json.loads and return an empty dict if data is invalid
    """
    assert error_handling in [i.value for i in ErrorHandlingEnum]
    if error_handling == ErrorHandlingEnum.FAIL:
        output = json.loads(str_to_check)
    else:
        try:
            output = json.loads(str_to_check)
        except (TypeError, ValueError):
            logging.warning("Invalid JSON: '" + str(str_to_check) + "'")
            output = {}
    return output


def fail_or_warn_on_row(
    api_exceptions: Union[Exception, Tuple[Exception]] = API_EXCEPTIONS,
    error_handling: AnyStr = ErrorHandlingEnum.FAIL.value,
    verbose: bool = False
) -> Callable:
    """
    Decorate an API calling function to:
    - ensure it has a 'row' parameter which is a dict (BATCH *not* supported)
    - return the row with a 'raw_result' key containing the function result
    - handles errors from the function with two methods:
        * (fail - default) fail if there is an error
        * (warn) do not fail on a list of API-related exceptions, just log it
        and return the row with a new 'error' key
     """
    assert error_handling in [i.value for i in ErrorHandlingEnum]

    def inner_decorator(func):
        if "row" not in inspect.getfullargspec(func).args:
            raise ValueError("Function must have 'row' as first parameter.")

        @wraps(func)
        def wrapped(row, *args, **kwargs):
            if not isinstance(row, dict):
                raise ValueError(
                    "The 'row' parameter must be a dict or a list of dict.")
            response_key = generate_unique("raw_response", row.keys())
            error_message_key = generate_unique("error_message", row.keys())
            error_type_key = generate_unique("error_type", row.keys())
            error_raw_key = generate_unique("error_raw", row.keys())
            new_keys = [
                response_key, error_message_key,
                error_type_key, error_raw_key
            ]
            if error_handling == ErrorHandlingEnum.FAIL.value:
                row[response_key] = func(row=row, *args, **kwargs)
                return row
            elif error_handling == ErrorHandlingEnum.WARN.value:
                for k in new_keys:
                    row[k] = ''
                try:
                    row[response_key] = func(row=row, *args, **kwargs)
                    return row
                except api_exceptions as e:
                    error_str = str(e)
                    logging.warning(error_str)
                    module = str(inspect.getmodule(e).__name__)
                    class_name = str(type(e).__qualname__)
                    error_type = module + "." + class_name
                    error_raw = str(e.args)
                    row[error_message_key] = error_str
                    row[error_type_key] = error_type
                    if verbose:
                        row[error_raw_key] = error_raw
                    else:
                        del row[error_raw_key]
                    return row

        return wrapped

    return inner_decorator


def fail_or_warn_on_batch():
    # TODO write it ;)
    # May need to write cloud-specific decorator
    return None


def api_parallelizer(
    input_df: pd.DataFrame,
    api_call_function: Callable,
    parallel_workers: int = 5,
    api_support_batch: bool = False,
    batch_size: int = 10,
    verbose: bool = False,
    **api_call_function_kwargs
) -> pd.DataFrame:
    """
    Apply an API call function in parallel to a pandas.DataFrame.
    The DataFrame is passed to the function as row dictionaries.
    Parallelism works by:
    - (default) sending multiple concurrent threads
    - if the API supports it, sending batches of row
    """
    df_iterator = (i[1].to_dict() for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    if api_support_batch:
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    response_col = generate_unique("raw_response", input_df.columns)
    error_message_col = generate_unique("error_message", input_df.columns)
    error_type_col = generate_unique("error_type", input_df.columns)
    output_schema = {
        **{response_col: str, error_message_col: str, error_type_col: str},
        **dict(input_df.dtypes)
    }
    if verbose:
        error_raw_col = generate_unique("error_raw", input_df.columns)
        output_schema = {**output_schema,  **{error_raw_col: str}}
    results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        futures = [
            pool.submit(
                fn=api_call_function, row=row, **api_call_function_kwargs)
            for row in df_iterator
        ]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            results.append(f.result())
    if api_support_batch:
        results = flatten(results)
    record_list = [
        {col: result.get(col) for col in output_schema.keys()}
        for result in results
    ]
    column_list = list(input_df.columns)
    column_list += [response_col, error_message_col, error_type_col]
    if verbose:
        column_list += [error_raw_col]
    output_df = pd.DataFrame.from_records(record_list) \
        .astype(output_schema) \
        .reindex(columns=column_list)
    return output_df
