# -*- coding: utf-8 -*-
import logging
from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku

from param_enums import ErrorHandlingEnum, OutputFormatEnum
from api_calling_utils import (
    build_unique_column_names, api_parallelizer, validate_column_input)
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role)
from dku_gcp_nlp import (
    DOCUMENT_TYPE, ENCODING_TYPE, DEFAULT_AXIS_NUMBER,
    get_client, format_named_entity_recognition)


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
service_account_key = api_configuration_preset.get("gcp_service_account_key")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language", '').replace("auto", '')
output_format = OutputFormatEnum[get_recipe_config().get('output_format')]
entity_sentiment = get_recipe_config().get('entity_sentiment', False)
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
column_prefix = "ner_api"
api_column_names = build_unique_column_names(input_df, column_prefix)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_named_entity_recognition(
    row, text_column, text_language, entity_sentiment
):
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == '':
        return('')
    else:
        document = language.types.Document(
            content=text, language=text_language, type=DOCUMENT_TYPE)
        if entity_sentiment:
            response = client.analyze_entity_sentiment(
                document=document, encoding_type=ENCODING_TYPE)
        else:
            response = client.analyze_entities(
                document=document, encoding_type=ENCODING_TYPE)
        return MessageToJson(response)


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_named_entity_recognition,
    parallel_workers=parallel_workers,  error_handling=error_handling,
    column_prefix=column_prefix, text_column=text_column,
    text_language=text_language, entity_sentiment=entity_sentiment
)

logging.info("Formatting API results...")
output_df = output_df.apply(
    func=format_named_entity_recognition, axis=DEFAULT_AXIS_NUMBER,
    response_column=api_column_names.response, output_format=output_format,
    error_handling=error_handling, column_prefix=column_prefix)
logging.info("Formatting API results: Done.")

output_dataset.write_with_schema(output_df)
