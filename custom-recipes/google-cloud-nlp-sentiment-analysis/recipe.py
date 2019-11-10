import dataiku
import logging
import copy
from google.cloud import language
from google.protobuf.json_format import MessageToJson
from dataiku.customrecipe import *
from dku_gcp_nlp import *

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[Google Cloud plugin] %(levelname)s - %(message)s')

recipe_config = get_recipe_config()
connection_info = recipe_config.get("connection_info")
input_text_col = recipe_config.get("input_text_col")
sentiment_scale = recipe_config.get("sentiment_scale", "ternary")
output_magnitude = recipe_config.get("output_magnitude")
output_probabilities = recipe_config.get("output_probabilities")
output_detected_language = recipe_config.get("output_detected_language")
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
output_schema.append({"name": "predicted_sentiment", "type": "string"})
if output_magnitude:
    output_schema.append({"name": "magnitude", "type": "string"})
if output_probabilities:
    output_schema.append({"name": "predicted_probability", "type": "double"})
if output_detected_language:
    output_schema.append({"name": "detected_language", "type": "string"})
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for input_row in input_dataset.iter_rows():
        output_row = dict(input_row)

        input_text = input_row[input_text_col]
        if input_text is not None and len(input_text):
            document = language.types.Document(content=input_text, type=language.enums.Document.Type.PLAIN_TEXT)
            response = client.analyze_sentiment(document=document, encoding_type='UTF32')

            result = json.loads(MessageToJson(response))

            if result["documentSentiment"].get("score") is None:
                logging.warn("API did not return sentiment")
            else:
                output_row["predicted_sentiment"] = get_sentiment(result["documentSentiment"].get("score"), sentiment_scale)
            if output_magnitude:
                output_row["magnitude"] = result["documentSentiment"].get("magnitude")
            if output_detected_language:
                output_row["detected_language"] = result["language"]
            if should_output_raw_results:
                output_row["raw_results"] = json.dumps(result)
        writer.write_row_dict(output_row)
