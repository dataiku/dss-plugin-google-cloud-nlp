# -*- coding: utf-8 -*-
# Note this implementation support batch while Google APIs don't (the code is common with AWS)
import itertools
import more_itertools
from typing import Callable, List, Dict, Tuple, Type, AnyStr, Union
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm as tqdm_auto
import math
import numpy as np
import pandas as pd


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return(new_name)
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def do_sthg_stupid(x):
    x["foo"] = "bar"
    return(x)


def do_sthg_stupid_in_batch(x):
    for i in x:
        i["foo"] = "bar"
    return(x)


def parallel_api_caller(input_df:            pd.DataFrame,
                        api_call_function:   Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
                        output_schema:       Dict[AnyStr, Type],
                        parallel_processes:  int = 10,
                        api_support_batch:   bool = False,
                        batch_size:          int = 20,
                        timeout:             int = 30000
                        ) -> pd.DataFrame:
    # TODO add input_column parameter to feed to api_call_function
    # TODO add logic to automatically reroute return of api_call_function to 2 columns: response and error
    # TODO think about is it really our job to care about output_schema? or just route to response/error
    # and then let the user wrangle the wanted format out of the JSON response?
    # TODO after all should we add error handling here?
    df_iterator = (i[1] for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    if api_support_batch:
        df_iterator = more_itertools.chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    output_schema = {**output_schema, **dict(input_df.dtypes)}
    with ProcessPoolExecutor(max_workers=parallel_processes) as pool:
        map_iterator = pool.map(
            api_call_function, df_iterator, timeout=timeout)
        results = list(tqdm_auto(map_iterator, total=len_iterator))

    if api_support_batch:
        results = more_itertools.flatten(results)
    record_list = [
        {col: result.get(col) for col in output_schema.keys()} for result in results]
    output_df = pd.DataFrame.from_records(record_list).astype(output_schema)
    return(output_df)
