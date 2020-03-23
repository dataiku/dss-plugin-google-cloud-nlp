# -*- coding: utf-8 -*-
import itertools
import logging
import time
import math
import numpy as np
import pandas as pd

from typing import Callable, List, Dict, Tuple, Type, AnyStr, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

API_RATE_LIMIT_PERIOD = 60  # 1 minute
API_RATE_LIMIT_QUOTA = 600  # 600 calls per period


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return(new_name)
            break
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


@on_exception(expo, RateLimitException, max_time=API_RATE_LIMIT_PERIOD, max_tries=5)
@limits(calls=API_RATE_LIMIT_QUOTA, period=API_RATE_LIMIT_PERIOD)
def api_call_function_wrapper(api_call_function, row, error_handling, **api_call_function_kwargs):
    if error_handling not in ["warn", "fail"]:
        logging.error(
            "Error handling parameter can be either 'warn' or 'fail'.")
    for new_key in ["response", "error"]:
        row[generate_unique(new_key, row.keys())] = ''
        try:
            row["response"] = api_call_function(
                row=row, **api_call_function_kwargs)
        except Exception as e:
            if error_handling == "warn":
                row["error"] = str(e)
                logging.warning(row["error"])
                return(row)
            else:
                raise e
    return(row)


def api_parallelizer(input_df:            pd.DataFrame,
                    api_call_function:   Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
                    parallel_processes:  int = 10,
                    api_support_batch:   bool = False,
                    batch_size:          int = 20,
                    error_handling:      AnyStr = "fail",
                    **api_call_function_kwargs
                    ) -> pd.DataFrame:
    df_iterator = (i[1] for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    if api_support_batch:
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    output_schema = {**{"response": str, "error": str},
                     **dict(input_df.dtypes)}
    results = []
    with ProcessPoolExecutor(max_workers=parallel_processes) as pool:
        futures = [pool.submit(fn=api_call_function_wrapper,
                               api_call_function=api_call_function,
                               row=row,
                               error_handling=error_handling,
                               **api_call_function_kwargs)
                   for row in df_iterator]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            results.append(f.result())
    if api_support_batch:
        results = flatten(results)
    record_list = [
        {col: result.get(col) for col in output_schema.keys()} for result in results]
    output_df = pd.DataFrame.from_records(record_list).astype(output_schema)
    return(output_df)
