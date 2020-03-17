# -*- coding: utf-8 -*-
# Note this implementation support batch while Google APIs don't (the code is common with AWS)
import itertools
import math
import time
from tqdm.contrib.concurrent import process_map
from typing import Callable, List, Iterator, Tuple, Type, Union


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return(new_name)
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return(itertools.zip_longest(*args, fillvalue=fillvalue))


def do_sthg_stupid(x):
    # TODO error handling and raw response output
    output_dictionary = x[1]
    output_dictionary["foo"] = "bar"
    return(output_dictionary)


def do_sthg_stupid_in_batch(x):
     # TODO error handling and raw response output
    output_dictionary = [dict(i[1]) for i in x if i is not None]
    for i in output_dictionary:
        i["foo"] = "bar"
    return(output_dictionary)


def parallel_api_caller(input_df:            pd.DataFrame,
                        api_call_function:   Callable[[Union[Tuple, List[Tuple]]], Union[Tuple, List[Tuple]]],
                        output_schema:       Dict[AnyStr, Type],
                        parallel_processes:  int = 10,
                        api_support_batch:   bool = False,
                        batch_size:          int = 20,
                        **kwargs
                        ) -> pd.DataFrame:
    # TODO test chunksize parameter directly on concurrent.futures
    # TODO see if using concurrent.futures directly is not better, and then use submit?
    df_iterator = input_df.iterrows()
    len_iterator = len(input_df.index)
    if api_support_batch:
        df_iterator = grouper(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)

    results = process_map(api_call_function, df_iterator,
                          total=len_iterator, max_workers=parallel_processes)
    if api_support_batch:
        results = list(itertools.chain(*results))
    output_df = pd.DataFrame.from_records(
        [{col: result.get(col) for col in output_schema.keys()}
         for result in results]
    ).astype(output_schema)
    return(output_df)
