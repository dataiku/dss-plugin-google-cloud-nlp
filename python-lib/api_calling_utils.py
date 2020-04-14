# -*- coding: utf-8 -*-
import logging
import inspect
import math
import json

from enum import Enum
from collections import OrderedDict
from typing import Callable, AnyStr, List, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class ErrorHandlingEnum(Enum):
    LOG = "log"
    RAISE = "raise"


API_COLUMN_LIST = ["response", "error_message", "error_type", "error_raw"]

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


def generate_unique(
    name: AnyStr,
    existing_names: List,
    prefix: AnyStr = None
) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number.
    Can also add an optional prefix.
    """
    if prefix is not None:
        new_name = prefix + "_" + name
    else:
        new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def safe_json_loads(
    str_to_check: AnyStr,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    """
    Wrap json.loads with an additional parameter to handle errors:
    - 'RAISE' to use json.loads, which fails on invalid data
    - 'LOG' to try json.loads and return an empty dict if data is invalid
    """
    if error_handling == ErrorHandlingEnum.RAISE:
        output = json.loads(str_to_check)
    else:
        try:
            output = json.loads(str_to_check)
        except (TypeError, ValueError):
            logging.warning("Invalid JSON: '" + str(str_to_check) + "'")
            output = {}
    return output


def fail_or_warn_row(
    api_call_function: Callable,
    api_column_dict: Dict,
    row: Dict,
    api_exceptions: Union[Exception, Tuple[Exception]] = API_EXCEPTIONS,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = False,
    **api_call_function_kwargs
) -> Callable:
    """
    Wraps an API calling function to:
    - ensure it has a 'row' parameter which is a dict (BATCH *not* supported)
    - return the row with a 'raw_result' key containing the function result
    - handles errors from the function with two methods:
        * (RAISE) fail if there is an error
        * (LOG - default) do not fail on API-related exceptions, just log it
        and return the row with a new 'error' key
    """
    if error_handling == ErrorHandlingEnum.RAISE:
        row[api_column_dict["response"]] = api_call_function(
            row=row, **api_call_function_kwargs)
        return row
    else:
        for k in api_column_dict.values():
            row[k] = ''
        try:
            row[api_column_dict["response"]] = api_call_function(
                row=row, **api_call_function_kwargs)
        except api_exceptions as e:
            logging.warning(str(e))
            module = str(inspect.getmodule(e).__name__)
            error_name = str(type(e).__qualname__)
            row[api_column_dict["error_message"]] = str(e)
            row[api_column_dict["error_type"]] = module + "." + error_name
            row[api_column_dict["error_raw"]] = str(e.args)
        return row


def fail_or_warn_batch():
    # TODO write it ;)
    return None


def initialize_api_column_names(
    input_df: pd.DataFrame,
    column_prefix: AnyStr = "api"
) -> OrderedDict:
    """
    Helper function to the "api_parallelizer" main function.
    Initializes a dictionary of column names from API_COLUMN_LIST,
    adding a prefix and a number suffix to make them unique.
    """
    api_column_dict = OrderedDict(
        (k, generate_unique(k, input_df.columns, column_prefix))
        for k in API_COLUMN_LIST)
    return api_column_dict


def convert_api_results_to_df(
    input_df: pd.DataFrame,
    api_results: List[Dict],
    api_column_dict: Dict,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Helper function to the "api_parallelizer" main function.
    Combine API results (list of dict) with input dataframe,
    and convert it to a dataframe.
    """
    if not verbose:
        del api_column_dict["error_raw"]
    output_schema = {
        **{v: str for v in api_column_dict.values()},
        **dict(input_df.dtypes)}
    record_list = [
        {col: result.get(col) for col in output_schema.keys()}
        for result in api_results]
    column_list = list(input_df.columns) + list(api_column_dict.values())
    output_df = pd.DataFrame.from_records(record_list) \
        .astype(output_schema) \
        .reindex(columns=column_list)
    assert len(output_df.index) == len(input_df.index)
    return(output_df)


def api_parallelizer(
    input_df: pd.DataFrame,
    api_call_function: Callable,
    parallel_workers: int = 5,
    api_support_batch: bool = False,
    batch_size: int = 10,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    column_prefix: AnyStr = "api",
    api_exceptions: Union[Exception, Tuple[Exception]] = API_EXCEPTIONS,
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
    api_column_dict = initialize_api_column_names(input_df, column_prefix)
    pool_kwargs = api_call_function_kwargs.copy()
    more_kwargs = [
        'api_call_function', 'error_handling',
        'api_exceptions', 'api_column_dict']
    for k in more_kwargs:
        pool_kwargs[k] = locals()[k]
    for k in ["fn", "row", "batch"]:  # Reserved pool keyword arguments
        pool_kwargs.pop(k, None)
    api_results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        if api_support_batch:
            futures = [
                pool.submit(fail_or_warn_batch, batch=batch, **pool_kwargs)
                for batch in df_iterator]
        else:
            futures = [
                pool.submit(fail_or_warn_row, row=row, **pool_kwargs)
                for row in df_iterator]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            api_results.append(f.result())
    if api_support_batch:
        api_results = flatten(api_results)
    output_df = convert_api_results_to_df(
        input_df, api_results, api_column_dict, verbose)
    return output_df
