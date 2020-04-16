from enum import Enum


class ErrorHandlingEnum(Enum):
    log = "Log"
    fail = "Fail"


class OutputFormatEnum(Enum):
    single_column = "Single JSON column"
    multiple_columns = "Multiple columns"
