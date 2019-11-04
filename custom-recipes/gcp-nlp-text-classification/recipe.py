import dataiku
import logging
import copy
from google.cloud import language
from google.protobuf.json_format import MessageToJson
from dataiku.customrecipe import *
from misc_helpers import get_credentials

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[Google Cloud plugin] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

connection_info = get_recipe_config().get("connection_info")
input_text_col = get_recipe_config().get("input_text_col")

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

credentials = get_credentials(connection_info)
client = language.LanguageServiceClient(credentials=credentials)

#==============================================================================
# RUN
#==============================================================================

output_schema = copy.deepcopy(input_schema)
output_schema.append({"name": "categories", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for input_row in input_dataset.iter_rows():
        output_row = dict(input_row)
        input_text = input_row[input_text_col]
        if input_text is not None and len(input_text):
            document = language.types.Document(content=input_text, type=language.enums.Document.Type.PLAIN_TEXT)
            categories = None
            try:
                response = client.classify_text(document=document)
                resp_json = json.loads(MessageToJson(response))
                categories = resp_json.get("categories")
            except Exception as e:
                # TODO re-raise if not a token thing
                pass
            if categories is not None:
                output_row["categories"] = json.dumps(categories)
        writer.write_row_dict(output_row)
