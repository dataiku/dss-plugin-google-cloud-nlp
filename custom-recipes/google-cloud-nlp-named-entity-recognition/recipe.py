# -*- coding: utf-8 -*-
import logging

from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku
from api_calling_utils import (
    ErrorHandlingEnum, initialize_api_column_names, api_parallelizer
)
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role
)
from dku_gcp_nlp import (
    DOCUMENT_TYPE, ENCODING_TYPE,
    get_client, format_named_entity_recognition
)


# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[Google Cloud NLP plugin] %(levelname)s - %(message)s'
)

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
gcp_service_account_key = api_configuration_preset.get(
    "gcp_service_account_key")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language", '').replace("auto", '')
output_format = get_recipe_config().get('output_format')
entity_sentiment = get_recipe_config().get('entity_sentiment')
error_handling = ErrorHandlingEnum[get_recipe_config().get('error_handling')]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column.")
if text_column not in input_columns_names:
    raise ValueError(
        "Column '{}' is not present in the input dataset.".format(text_column)
    )

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
client = get_client(gcp_service_account_key)
column_prefix = "ner_api"
api_column_dict = initialize_api_column_names(input_df, column_prefix)


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_named_entity_recognition(
    row, text_column, text_language, entity_sentiment
):
    text = row[text_column]
    if not isinstance(text, str) or text.strip() == '':
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

output_df = output_df.apply(
    func=format_named_entity_recognition, axis=1,
    response_column=api_column_dict["response"], output_format=output_format,
    error_handling=error_handling, column_prefix=column_prefix)

output_dataset.write_with_schema(output_df)
