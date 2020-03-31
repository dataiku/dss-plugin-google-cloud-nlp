# -*- coding: utf-8 -*-
import logging

from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language
from google.protobuf.json_format import MessageToJson

import dataiku
from common import generate_unique, fail_or_warn_on_row, api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role,
    get_output_names_for_role
)
from dku_gcp_nlp import (
    DOCUMENT_TYPE, get_client, format_text_classification
)


# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[Google Cloud NLP plugin] %(levelname)s - %(message)s'
)

api_configuration_preset = get_recipe_config().get("api_configuration_preset", {})
gcp_service_account_key = api_configuration_preset.get(
    "gcp_service_account_key")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language", '').replace("auto", '')
output_format = get_recipe_config().get('output_format')
num_categories = int(get_recipe_config().get('num_categories'))
error_handling = get_recipe_config().get('error_handling')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
response_column = generate_unique("raw_response", input_df.columns)
client = get_client(gcp_service_account_key)


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
@fail_or_warn_on_row(error_handling=error_handling)
def call_api_text_classification(row, text_column, text_language=None):
    text = row[text_column]
    if not isinstance(text, str) or text.strip() == '':
        return('')
    else:
        document = language.types.Document(
            content=text, language=text_language, type=DOCUMENT_TYPE)
        response = client.classify_text(
            document=document)
        return MessageToJson(response)


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_text_classification,
    text_column=text_column, text_language=text_language,
    parallel_workers=parallel_workers)

output_df = output_df.apply(
    func=format_text_classification, axis=1,
    response_column=response_column, output_format=output_format,
    num_categories=num_categories, error_handling=error_handling)

output_dataset.write_with_schema(output_df)
