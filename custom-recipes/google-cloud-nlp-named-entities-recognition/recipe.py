# -*- coding: utf-8 -*-
import logging

from ratelimit import limits, RateLimitException
from retry import retry
from google.cloud import language

import dataiku
from dataiku.customrecipe import *

from dku_gcp_nlp import *
from common import *


# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(level=logging.INFO,
                    format='[Google Cloud NLP plugin] %(levelname)s - %(message)s')

cloud_configuration_preset = get_recipe_config().get("cloud_configuration_preset", {})
gcp_service_account_key = cloud_configuration_preset.get("gcp_service_account_key")
api_quota_rate_limit = cloud_configuration_preset.get("api_quota_rate_limit")
api_quota_period = cloud_configuration_preset.get("api_quota_period")
parallel_workers = cloud_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language", '').replace("auto", '')
output_format = get_recipe_config().get('output_format')
error_handling = get_recipe_config().get('error_handling')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
client = get_client(gcp_service_account_key)

@retry((RateLimitException, ConnectionError), delay=api_quota_period, tries=10)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
@fail_or_warn_on_row(error_handling=ErrorHandlingEnum.FAIL)
def call_api_named_entity_recognition(row, text_column, text_language=None):
    text = row[text_column]
    if not isinstance(text, str):
        return('')
    else:
        document = language.types.Document(
            content=text, language=text_language, type=DOCUMENT_TYPE)
        response = client.analyze_entities(
            document=document,
            encoding_type=ENCODING_TYPE
        )
        return MessageToJson(response)


output_df = api_parallelizer(input_df=input_df, api_call_function=call_api_named_entity_recognition,
                             text_column=text_column, text_language=text_language,
                             parallel_workers=parallel_workers)

output_dataset.write_with_schema(output_df)
