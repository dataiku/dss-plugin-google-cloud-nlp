# -*- coding: utf-8 -*-
import dataiku
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
text_language = get_recipe_config().get("language")
if text_language == "auto":
    text_language = None
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


def call_api_named_entity_recognition(row, text_column, text_language=None):
    document = language.types.Document(
        content=row[text_column], language=text_language, type=DOCUMENT_TYPE)
    response = client.analyze_sentiment(
        document=document,
        encoding_type=ENCODING_TYPE
    )
    return(MessageToJson(response))


output_df = api_parallelizer(input_df, call_api_named_entity_recognition,
                             text_column=text_column, text_language=text_language, error_handling='warn')

output_dataset.write_with_schema(output_df)
