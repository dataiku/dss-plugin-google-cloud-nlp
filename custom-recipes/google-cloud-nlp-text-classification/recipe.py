# -*- coding: utf-8 -*-
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

from google_nlp_api_client import (
    DOCUMENT_TYPE,
    API_EXCEPTIONS,
    get_client,
)
from plugin_io_utils import (
    ErrorHandlingEnum,
    validate_column_input,
    set_column_description,
)
from api_parallelizer import api_parallelizer
from google_nlp_api_formatting import TextClassificationAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
if api_configuration_preset is None or api_configuration_preset == {}:
    raise ValueError("Please specify an API configuration preset")
service_account_key = api_configuration_preset.get("gcp_service_account_key")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language", "").replace("auto", "")
num_categories = int(get_recipe_config().get("num_categories"))
error_handling = ErrorHandlingEnum[get_recipe_config().get("error_handling")]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col["name"] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)
input_df = input_dataset.get_dataframe()
client = get_client(service_account_key)
column_prefix = "text_classif_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_text_classification(
    row: Dict, text_column: AnyStr, text_language: AnyStr
) -> AnyStr:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == "":
        return ""
    else:
        document = language.types.Document(
            content=text, language=text_language, type=DOCUMENT_TYPE
        )
        response = client.classify_text(document=document)
        return MessageToJson(response)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_text_classification,
    api_exceptions=API_EXCEPTIONS,
    column_prefix=column_prefix,
    parallel_workers=parallel_workers,
    error_handling=error_handling,
    text_column=text_column,
    text_language=text_language,
)

api_formatter = TextClassificationAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    num_categories=num_categories,
    error_handling=error_handling,
)
output_df = api_formatter.format_df(df)

output_dataset.write_with_schema(output_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=api_formatter.column_description_dict,
)
