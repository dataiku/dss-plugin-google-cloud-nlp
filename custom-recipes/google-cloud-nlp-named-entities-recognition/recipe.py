import dataiku
import logging
import copy
import json
from google.cloud import language
from google.protobuf.json_format import MessageToJson
from dataiku.customrecipe import *
from dku_gcp_nlp import *

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[Google Cloud plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get("connection_info")
input_text_col = get_recipe_config().get("input_text_col")
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = copy.deepcopy(input_schema)
output_schema.append({"name": "entities", "type": "string"})
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for input_row in input_dataset.iter_rows():
        output_row = dict(input_row)
        input_text = input_row[input_text_col]
        if input_text is not None and len(input_text):
            document = language.types.Document(content=input_text, type=language.enums.Document.Type.PLAIN_TEXT)
            response = client.analyze_entities(document=document, encoding_type='UTF32')
            entities = [json.loads(MessageToJson(e)) for e in response.entities]
            output_row["entities"] = json.dumps(entities)
        writer.write_row_dict(output_row)
