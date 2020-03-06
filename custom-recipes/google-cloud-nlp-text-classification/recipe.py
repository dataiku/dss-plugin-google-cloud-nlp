import dataiku
import logging
import time
import json
from google.cloud import language as nlp
from dataiku.customrecipe import *
from dku_gcp_nlp import *
from common import *

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[Google Cloud NLP plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get("connection_info")
text_column = get_recipe_config().get("text_column")
language = get_recipe_config().get("language")
remove_prefix = get_recipe_config().get("remove_prefix")
output_multiple_categories = get_recipe_config().get("output_multiple_categories")
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]
categories_column = generate_unique('categories', input_columns_names)
category_column = generate_unique('category', input_columns_names)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

#==============================================================================
# RUN
#==============================================================================

input_df = input_dataset.get_dataframe()

@with_original_indices
def classify(text_list):
    client = get_client(connection_info)
    logging.info("request: %d characters" % (sum([len(t) for t in text_list])))
    start = time.time()
    document = nlp.types.Document(content = text_list[0], type = nlp.enums.Document.Type.PLAIN_TEXT, language = language)
    response = client.classify_text(document=document) #, encoding_type='UTF32'
    logging.info("request took %.3fs" % (time.time() - start))
    return response


for batch in run_by_batch(classify, input_df, text_column, batch_size=BATCH_SIZE, parallelism=PARALLELISM):
    raw_results, original_indices = batch
    j = original_indices[0]
    output = format_classification_results(raw_results, remove_prefix)
    if output_multiple_categories:
        input_df.set_value(j, categories_column, json.dumps(output['categories']) if len(output['categories']) else '')
    else:
        input_df.set_value(j, category_column, output['categories'][0] if len(output['categories']) else '')
    if should_output_raw_results and output['raw_results']:
        input_df.set_value(j, 'raw_results', json.dumps(output['raw_results']))

output_dataset.write_with_schema(input_df)
