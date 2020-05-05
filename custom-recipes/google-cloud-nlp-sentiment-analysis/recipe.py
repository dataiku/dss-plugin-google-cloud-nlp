# -*- coding: utf-8 -*-
import logging
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku

from plugin_io_utils import (
    COLUMN_DESCRIPTION_DICT, ErrorHandlingEnum, build_unique_column_names,
    generate_unique, validate_column_input, set_column_description)
from api_parallelizer import api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role)
from cloud_api import (
    DOCUMENT_TYPE, ENCODING_TYPE, APPLY_AXIS,
    get_client, format_sentiment_analysis, move_api_columns_to_end)


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
text_language = get_recipe_config().get("language", '').replace("auto", '')
sentiment_scale = get_recipe_config().get('sentiment_scale')
error_handling = ErrorHandlingEnum[get_recipe_config().get('error_handling')]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)
input_df = input_dataset.get_dataframe()
client = get_client(service_account_key)
column_prefix = "sentiment_api"
api_column_names = build_unique_column_names(input_df, column_prefix)
sentiment_score_column = generate_unique(
    "score", input_columns_names, column_prefix)
sentiment_score_scaled_column = generate_unique(
    "score_scaled", input_columns_names, column_prefix)
sentiment_magnitude_column = generate_unique(
    "magnitude", input_columns_names, column_prefix)

# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_sentiment_analysis(
    row: Dict, text_column: AnyStr, text_language: AnyStr
) -> AnyStr:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == '':
        return ''
    else:
        document = language.types.Document(
            content=text, language=text_language, type=DOCUMENT_TYPE)
        response = client.analyze_sentiment(
            document=document, encoding_type=ENCODING_TYPE)
        return MessageToJson(response)


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_sentiment_analysis,
    parallel_workers=parallel_workers, error_handling=error_handling,
    column_prefix=column_prefix, text_column=text_column,
    text_language=text_language)

logging.info("Formatting API results...")
output_df = output_df.apply(
    func=format_sentiment_analysis, axis=APPLY_AXIS,
    response_column=api_column_names.response, sentiment_scale=sentiment_scale,
    error_handling=error_handling, column_prefix=column_prefix)
output_df = move_api_columns_to_end(output_df, api_column_names)
logging.info("Formatting API results: Done.")

output_dataset.write_with_schema(output_df)
column_description_dict = {
    v: COLUMN_DESCRIPTION_DICT[k]
    for k, v in api_column_names._asdict().items()}
column_description_dict[sentiment_score_column] = \
    "Sentiment score from the API in numerical format between -1 and 1"
column_description_dict[sentiment_score_scaled_column] = \
    "Scaled sentiment score according to the “Sentiment scale” parameter"
column_description_dict[sentiment_magnitude_column] = \
    "Magnitude score from the API indicating the strength of emotion " + \
    "(both positive and negative) between 0 and +Inf"
set_column_description(output_dataset, column_description_dict)
