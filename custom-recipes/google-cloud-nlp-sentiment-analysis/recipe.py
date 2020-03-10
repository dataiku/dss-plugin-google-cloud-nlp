import dataiku
import logging
import time
from google.cloud import language as nlp
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
sentiment_scale = get_recipe_config().get("sentiment_scale", "ternary")
language = get_recipe_config().get("language")
if language == "auto":
    language = None
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]
predicted_sentiment_column = generate_unique(
    'predicted_sentiment', input_columns_names)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()


@with_original_indices
def detect_sentiment(text_list):
    client = get_client(cloud_credentials_preset)
    logging.info("request: %d characters" % (sum([len(t) for t in text_list])))
    start = time.time()
    document = nlp.types.Document(
        content=text_list[0], type=nlp.enums.Document.Type.PLAIN_TEXT, language=language)
    response = client.analyze_sentiment(
        document=document, encoding_type='UTF32')
    logging.info("request took %.3fs" % (time.time() - start))
    return response


for batch in run_by_batch(detect_sentiment, input_df, text_column, batch_size=BATCH_SIZE, parallelism=PARALLELISM):
    raw_results, original_indices = batch
    j = original_indices[0]
    output = format_sentiment_results(raw_results, sentiment_scale)
    input_df.set_value(j, predicted_sentiment_column,
                       output['predicted_sentiment'])
    if should_output_raw_results and output['raw_results']:
        input_df.set_value(j, 'raw_results', json.dumps(output['raw_results']))

output_dataset.write_with_schema(input_df)
