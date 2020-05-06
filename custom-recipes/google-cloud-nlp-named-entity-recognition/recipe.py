# -*- coding: utf-8 -*-
import logging
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku

from plugin_io_utils import (
    ErrorHandlingEnum, build_unique_column_names,
    validate_column_input, set_column_description)
from api_parallelizer import api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role)
from api_formatting import (
    DOCUMENT_TYPE, ENCODING_TYPE, EntityTypeEnum,
    get_client, format_df_named_entity_recognition,
    compute_column_description_named_entity_recognition)


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
entity_sentiment = get_recipe_config().get('entity_sentiment', False)
error_handling = ErrorHandlingEnum[get_recipe_config().get('error_handling')]
entity_types = [
    EntityTypeEnum[i] for i in get_recipe_config().get("entity_types", [])]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)
input_df = input_dataset.get_dataframe()
client = get_client(service_account_key)
column_prefix = "entity_api"
api_column_names = build_unique_column_names(input_df, column_prefix)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_named_entity_recognition(
    row: Dict, text_column: AnyStr, text_language: AnyStr,
    entity_sentiment: bool
) -> AnyStr:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == '':
        return ''
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
    parallel_workers=parallel_workers, error_handling=error_handling,
    column_prefix=column_prefix, text_column=text_column,
    text_language=text_language, entity_sentiment=entity_sentiment)

output_df = format_df_named_entity_recognition(
    df=output_df, api_column_names=api_column_names, entity_types=entity_types,
    column_prefix=column_prefix, error_handling=error_handling)
column_description_dict = compute_column_description_named_entity_recognition(
    df=input_df, api_column_names=api_column_names,
    column_prefix=column_prefix)

output_dataset.write_with_schema(output_df)
set_column_description(output_dataset, column_description_dict)
