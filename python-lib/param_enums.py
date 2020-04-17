from enum import Enum


class ErrorHandlingEnum(Enum):
    LOG = "Log"
    FAIL = "Fail"


class OutputFormatEnum(Enum):
    SINGLE_COLUMN = "Single JSON column"
    MULTIPLE_COLUMNS = "Multiple columns"
