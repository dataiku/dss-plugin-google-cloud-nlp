# -*- coding: utf-8 -*-
import json
import logging
from google.cloud import language
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

from common import generate_unique, safe_json_loads

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DOCUMENT_TYPE = language.enums.Document.Type.PLAIN_TEXT
ENCODING_TYPE = language.enums.EncodingType.UTF8

NAMED_ENTITY_TYPES = [
    'UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
    'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER',
    'PHONE_NUMBER', 'ADDRESS', 'DATE', 'NUMBER', 'PRICE'
]

# ==============================================================================
# FUNCTION DEFINITION
# ==============================================================================


def get_client(gcp_service_account_key=None):
    """
    Get a Google Natural Language API client from the service account key.
    """
    if gcp_service_account_key is None:
        return language.LanguageServiceClient()
    try:
        credentials = json.loads(gcp_service_account_key)
    except Exception as e:
        logging.error(e)
        raise ValueError("GCP service account key is not valid JSON.")
    credentials = service_account.Credentials.from_service_account_info(
        credentials)
    if hasattr(credentials, 'service_account_email'):
        logging.info("GCP service account loaded with email: %s" %
                     credentials.service_account_email)
    else:
        logging.info("Credentials loaded")
    return language.LanguageServiceClient(credentials=credentials)


def format_named_entity_recognition(row, response_column,
                                    output_format, error_handling):
    raw_response = row[response_column]
    response, valid_response = safe_json_loads(raw_response, error_handling)
    if output_format == "single_column":
        entity_column = generate_unique("entities", row.keys())
        row[entity_column] = response.get("entities", '')
    else:
        entities = response.get("entities", [])
        for n in NAMED_ENTITY_TYPES:
            entity_type_column = generate_unique(
                "entity_type_" + n.lower(), row.keys())
            row[entity_type_column] = [
                e for e in entities if e.get("type", '') == n
            ]
    return row


def format_sentiment(raw_results, scale="ternary"):
    #result = json.loads(MessageToJson(raw_results))
    result = raw_results
    output_row = {
        "raw_results": result,
        "predicted_sentiment": None
    }
    score = result.get("documentSentiment", {}).get("score")
    if score is not None:
        output_row['predicted_sentiment'] = scale_sentiment_score(score, scale)

    else:
        logging.warning("API did not return sentiment")
    return output_row


def scale_sentiment_score(score, scale):
    if scale == 'binary':
        return 'negative' if score < 0 else 'positive'
    elif scale == 'ternary':
        return 'negative' if score < -0.33 else 'positive' if score > 0.33 else 'neutral'
    elif scale == '1to5':
        if score < -0.66:
            return 'highly negative'
        elif score < -0.33:
            return 'negative'
        elif score < 0.33:
            return 'neutral'
        elif score < 0.66:
            return 'positive'
        else:
            return 'highly positive'
    elif scale == '0to1':
        return round((score+1)/2, 2)
    else:
        return round(score, 2)

def format_text_classification(raw_results):
    result = json.loads(MessageToJson(raw_results))
    output_row = dict()
    output_row['categories'] = [c['name']
                                for c in result.get('categories', [])]
    # if remove_prefix:
    #    output_row['categories'] = [c.split('/')[-1] for c in output_row['categories']]
    output_row["raw_results"] = result
    return output_row
