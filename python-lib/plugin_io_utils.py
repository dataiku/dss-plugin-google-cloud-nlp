import logging
import json
import dataiku

from enum import Enum
from typing import AnyStr, List, NamedTuple, Dict
from collections import namedtuple

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_COLUMN_NAMES = ["response", "error_message", "error_type", "error_raw"]
COLUMN_PREFIX = "api"
COLUMN_DESCRIPTION_DICT = {
    "response": "Raw response from the API in JSON format",
    "error_message": "Error message from the API",
    "error_type": "Error type (module and class name)",
    "error_raw": "Raw error from the API"}


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================

ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", API_COLUMN_NAMES)


class ErrorHandlingEnum(Enum):
    LOG = "Log"
    FAIL = "Fail"


class OutputFormatEnum(Enum):
    SINGLE_COLUMN = "Single JSON column"
    MULTIPLE_COLUMNS = "Multiple columns"


def generate_unique(
    name: AnyStr,
    existing_names: List,
    prefix: AnyStr = COLUMN_PREFIX
) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number.
    Can also add an optional prefix.
    """
    if prefix is not None:
        new_name = prefix + "_" + name
    else:
        new_name = name
    for j in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def build_unique_column_names(
    existing_names: List[AnyStr],
    column_prefix: AnyStr = COLUMN_PREFIX
) -> NamedTuple:
    """
    Helper function to the "api_parallelizer" main function.
    Initializes a named tuple of column names from ApiColumnNameTuple,
    adding a prefix and a number suffix to make them unique.
    """
    api_column_names = ApiColumnNameTuple(
        *[generate_unique(k, existing_names, column_prefix)
          for k in ApiColumnNameTuple._fields])
    return api_column_names


def safe_json_loads(
    str_to_check: AnyStr,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = False
) -> Dict:
    """
    Wrap json.loads with an additional parameter to handle errors:
    - 'FAIL' to use json.loads, which throws an exception on invalid data
    - 'LOG' to try json.loads and return an empty dict if data is invalid
    """
    if error_handling == ErrorHandlingEnum.FAIL:
        output = json.loads(str_to_check)
    else:
        try:
            output = json.loads(str_to_check)
        except (TypeError, ValueError):
            if verbose:
                logging.warning("Invalid JSON: '" + str(str_to_check) + "'")
            output = {}
    return output


def validate_column_input(
    column_name: AnyStr,
    column_list: List[AnyStr]
) -> None:
    """
    Validate that user input for column parameter is valid.
    """
    if column_name is None or len(column_name) == 0:
        raise ValueError(
            "You must specify the '{}' column.".format(column_name))
    if column_name not in column_list:
        raise ValueError(
            "Column '{}' is not present in the input dataset.".format(
                column_name))


def set_column_description(
    dataset: dataiku.Dataset,
    column_description_dict: Dict
) -> None:
    dataset_schema = dataset.read_schema()
    for col_info in dataset_schema:
        col_name = col_info.get('name')
        col_info['comment'] = column_description_dict.get(col_name)
    dataset.write_schema(dataset_schema)
