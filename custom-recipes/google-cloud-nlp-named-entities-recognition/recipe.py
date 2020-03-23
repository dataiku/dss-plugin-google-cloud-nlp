# -*- coding: utf-8 -*-
import dataiku
import json
from google.cloud import language
from dataiku.customrecipe import *
from dku_gcp_nlp import *
from common import *

# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(level=logging.INFO,
                    format='[Google Cloud NLP plugin] %(levelname)s - %(message)s')

cloud_credentials_preset = get_recipe_config().get("cloud_credentials_preset")
text_column = get_recipe_config().get("text_column")
language = get_recipe_config().get("language", "").replace("auto", "")
output_format = get_recipe_config().get('output_format')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
client = get_client(cloud_credentials_preset)


def call_api_named_entity_recognition(row, text_column, language):
    document = language.types.Document(
        content=row[text_column], language=language, type=DOCUMENT_TYPE)
    print("bar")
    response = client.analyze_sentiment(
        document=document,
        encoding_type=ENCODING_TYPE
    )
    print("success")
    return(MessageToJson(response))


output_df = api_parallelizer(input_df, call_api_named_entity_recognition,
                                text_column=text_column, language=language, error_handling='fail')

output_dataset.write_with_schema(output_df)
