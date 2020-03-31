# -*- coding: utf-8 -*-
import json
import logging
from google.cloud import language
from google.oauth2 import service_account
from typing import AnyStr, Dict, Union

from api_calling_utils import (
    ErrorHandlingEnum, generate_unique, safe_json_loads
)

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DOCUMENT_TYPE = language.enums.Document.Type.PLAIN_TEXT
ENCODING_TYPE = language.enums.EncodingType.UTF32

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
    client = language.LanguageServiceClient(credentials=credentials)
    return client


def format_named_entity_recognition(
    row: Dict,
    response_column: AnyStr,
    output_format: AnyStr = "multiple_columns",
    error_handling: AnyStr = ErrorHandlingEnum.FAIL.value
) -> Dict:
    """
    Format the API response for entity recognition to:
    - make sure response is valid JSON
    - expand results to multiple JSON columns (one by entity type)
    or put all entities as a list in a single JSON column
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
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
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ''
    return row


def format_sentiment_analysis(
    row: Dict,
    response_column: AnyStr,
    sentiment_scale: AnyStr = "ternary",
    error_handling: AnyStr = ErrorHandlingEnum.FAIL.value
) -> Dict:
    """
    Format the API response for sentiment analysis to:
    - make sure response is valid JSON
    - expand results to two score and magnitude columns
    - scale the score according to predefined categorical or numerical rules
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    sentiment_score_column = generate_unique(
        "sentiment_score", row.keys())
    sentiment_score_scaled_column = generate_unique(
        "sentiment_score_scaled", row.keys())
    sentiment_magnitude_column = generate_unique(
        "sentiment_magnitude", row.keys())
    sentiment = response.get("documentSentiment", {})
    if sentiment == {}:
        logging.warning("API did not return sentiment")
    sentiment_score = sentiment.get("score")
    magnitude_score = sentiment.get("magnitude")
    if sentiment_score is not None:
        row[sentiment_score_column] = float(sentiment_score)
        row[sentiment_score_scaled_column] = scale_sentiment_score(
            sentiment_score, sentiment_scale)
    else:
        row[sentiment_score_column] = None
        row[sentiment_score_scaled_column] = None
    if magnitude_score is not None:
        row[sentiment_magnitude_column] = float(magnitude_score)
    else:
        row[sentiment_magnitude_column] = None
    return row


def scale_sentiment_score(
    score: float,
    scale: AnyStr = 'ternary'
) -> Union[AnyStr, float]:
    """
    Scale the score according to predefined categorical or numerical rules
    """
    if scale == 'binary':
        return 'negative' if score < 0 else 'positive'
    elif scale == 'ternary':
        if score < -0.33:
            return 'negative'
        elif score > 0.33:
            return 'positive'
        else:
            return 'neutral'
    elif scale == 'quinary':
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
    elif scale == 'rescale_zero_to_one':
        return float((score+1.)/2)
    else:
        return float(score)


def format_text_classification(
    row: Dict,
    response_column: AnyStr,
    output_format: AnyStr = "multiple_columns",
    num_categories: int = 3,
    error_handling: AnyStr = ErrorHandlingEnum.FAIL.value
) -> Dict:
    """
    Format the API response for text classification to:
    - make sure response is valid JSON
    - expand results to multiple JSON columns (one by classification category)
    or put all categories as a list in a single JSON column
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    if output_format == "single_column":
        classification_column = generate_unique(
            "classification_categories", row.keys())
        row[classification_column] = response.get("categories", '')
    else:
        categories = response.get("categories", [])
        for n in range(num_categories):
            category_column = generate_unique(
                "classification_category_" + str(n), row.keys())
            confidence_column = generate_unique(
                "classification_category_" + str(n) + "_confidence",
                row.keys())
            if len(categories) > n:
                row[category_column] = categories[n].get("name", '')
                row[confidence_column] = categories[n].get("confidence")
            else:
                row[category_column] = ''
                row[confidence_column] = None
    return row
