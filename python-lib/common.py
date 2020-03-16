# -*- coding: utf-8 -*-
# Note this implementation support batch while Google APIs don't (the code is common with AWS)
from multiprocessing import Pool
import parmap
from functools import wraps
import time

# The Google NLP APIs do not support batch scoring
BATCH_SIZE = 1
PARALLELISM = 1


def with_original_indices(func):
    @wraps(func)
    def w(it):
        text_list, original_indices = it
        return(func(text_list), original_indices)
    return(w)


#def run_by_batch(func, input_df, text_column, batch_size, parallelism):
#    m = parmap.map(func,
#               _iter_non_empty_rows_batches(
#                   input_df, text_column, batch_size=batch_size),
#               pm_processes=PARALLELISM, pm_pbar=True
#               )
#    return(m)

# def run_by_batch(func, input_df, text_column, batch_size, parallelism):
#    iterator = p_uimap(func, _iter_non_empty_rows_batches(
#        input_df, text_column, batch_size=batch_size), num_cpus=0.2)
#    return(iterator)

def run_by_batch(func, input_df, text_column, batch_size, parallelism):
    with Pool(parallelism) as p:
        return(p.map(func, _iter_non_empty_rows_batches(input_df, text_column, batch_size=batch_size)))

def _iter_non_empty_rows_batches(input_df, text_column, batch_size):
    text_list = []
    original_indices = []
    for index, row in input_df.iterrows():
        row.fillna('')
        text = row.get(text_column, '')
        if isinstance(text, str) and text.strip() != '':
            text_list.append(text)
            original_indices.append(index)
        if len(text_list) == batch_size:
            yield text_list, original_indices
            text_list = []
            original_indices = []
    if len(text_list):
        yield text_list, original_indices


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return(new_name)
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")

from itertools import zip_longest
import concurrent
from typing import Callable, List, Dict, Type, Union
from concurrent.futures import ProcessPoolExecutor

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parallel_api_caller(input_df:            pd.DataFrame, 
                        api_call_function:   Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
                        output_schema:       Dict[AnyStr, Type],
                        parallel_process:    int  = 10,
                        api_support_batch:   bool = False,
                        batch_size:          int  = 20,
                        output_raw_response: bool = True,
                        error_handling:      str  = 'fail',
                       ) -> pd.DataFrame:
    
    if error_handling not in ['fail', 'warn']:
        logging.error("Error handling argument not supported. Choose 'fail' or 'warn'")
    
    df_row_iterator = input_df.itertuples(index = False)
    
    with ProcessPoolExecutor() as executor:
        if api_support_batch:
            df_chunk_iterator = grouper(df_row_iterator, batch_size)
            for rows in df_chunk_iterator:
                print("batch")
                print(rows)
                print("\n")
        else: # API does not support batch
            for row in df_row_iterator:
                print("not batch")
                print(row)
        
    return("foo")