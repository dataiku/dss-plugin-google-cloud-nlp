# Note this implementation support batch while Google APIs don't (the code is common with AWS)
from multiprocessing import Pool
from functools import wraps

BATCH_SIZE = 20
PARALLELISM = 20


def with_original_indices(func):
    @wraps(func)
    def w(it):
        text_list, original_indices = it
        return(func(text_list), original_indices)
    return(w)


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
    # TODO WHY THIS FUNCTION?
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return(new_name)
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")
