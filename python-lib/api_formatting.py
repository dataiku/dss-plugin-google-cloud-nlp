# -*- coding: utf-8 -*-
import json
import logging
import pandas as pd

from enum import Enum
from typing import AnyStr, Dict, List, Union, NamedTuple

from google.cloud import language
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.oauth2 import service_account

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT, generate_unique,
    safe_json_loads, ErrorHandlingEnum)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DOCUMENT_TYPE = language.enums.Document.Type.PLAIN_TEXT
ENCODING_TYPE = language.enums.EncodingType.UTF8

API_EXCEPTIONS = (GoogleAPICallError, RetryError)

API_SUPPORT_BATCH = False
BATCH_RESULT_KEY = None
BATCH_ERROR_KEY = None
BATCH_INDEX_KEY = None
BATCH_ERROR_MESSAGE_KEY = None
BATCH_ERROR_TYPE_KEY = None

VERBOSE = False


class EntityTypeEnum(Enum):
    ADDRESS = "Address"
    CONSUMER_GOOD = "Consumer good"
    DATE = "Date"
    EVENT = "Event"
    LOCATION = "Location"
    NUMBER = "Number"
    ORGANIZATION = "Organization"
    OTHER = "Other"
    PERSON = "Person"
    PRICE = "Price"
    UNKNOWN = "Unknown"
    WORK_OF_ART = "Work of art"

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


def scale_sentiment_score(
    score: float,
    scale: AnyStr = 'ternary'
) -> Union[AnyStr, float]:
    """
    Scale sentiment score according to categorical or numerical rules
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


def format_row_sentiment_analysis(
    row: Dict,
    response_column: AnyStr,
    sentiment_scale: AnyStr = "ternary",
    column_prefix: AnyStr = "sentiment_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    """
    Format Sentiment Analysis API response, row-by-row:
    - make sure response is valid JSON
    - expand results to two score and magnitude columns
    - scale the score according to predefined categorical or numerical rules
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    sentiment_score_column = generate_unique(
        "score", row.keys(), column_prefix)
    sentiment_score_scaled_column = generate_unique(
        "score_scaled", row.keys(), column_prefix)
    sentiment_magnitude_column = generate_unique(
        "magnitude", row.keys(), column_prefix)
    sentiment = response.get("documentSentiment", {})
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


def format_df_sentiment_analysis(
    df: pd.DataFrame,
    api_column_names: NamedTuple,
    sentiment_scale: AnyStr = "ternary",
    column_prefix: AnyStr = "sentiment_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> pd.DataFrame:
    """
    Format a DataFrame containing API responses for Sentiment Analysis
    """
    logging.info("Formatting API results...")
    df = df.apply(
        func=format_row_sentiment_analysis, axis=1,
        response_column=api_column_names.response,
        sentiment_scale=sentiment_scale, error_handling=error_handling,
        column_prefix=column_prefix)
    df = move_api_columns_to_end(df, api_column_names)
    logging.info("Formatting API results: Done.")
    return df


def compute_column_description_sentiment_analysis(
    df: pd.DataFrame,
    api_column_names: NamedTuple,
    column_prefix: AnyStr = "sentiment_api",
) -> Dict:
    """
    Compute dictionary of column descriptions for Sentiment Analysis API
    """
    column_description_dict = {
        v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
        for k, v in api_column_names._asdict().items()}
    sentiment_score_column = generate_unique(
        "score", df.keys(), column_prefix)
    sentiment_score_scaled_column = generate_unique(
        "score_scaled", df.keys(), column_prefix)
    sentiment_magnitude_column = generate_unique(
        "magnitude", df.keys(), column_prefix)
    column_description_dict[sentiment_score_column] = \
        "Sentiment score from the API in numerical format between -1 and 1"
    column_description_dict[sentiment_score_scaled_column] = \
        "Scaled sentiment score according to the “Sentiment scale” parameter"
    column_description_dict[sentiment_magnitude_column] = \
        "Magnitude score from the API indicating the strength of emotion " + \
        "(both positive and negative) between 0 and +Inf"
    return column_description_dict


def format_row_named_entity_recognition(
    row: Dict,
    response_column: AnyStr,
    entity_types: List,
    column_prefix: AnyStr = "entity_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    """
    Format Named Entity Recognition API response, row-by-row:
    - make sure response is valid JSON
    - expand results to multiple JSON columns (one by entity type)
    or put all entities as a list in a single JSON column
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    entities = response.get("entities", [])
    selected_entity_types = sorted([
        e.name for e in entity_types])
    for n in selected_entity_types:
        entity_type_column = generate_unique(
            "entity_type_" + n.lower(), row.keys(), column_prefix)
        row[entity_type_column] = [
            e.get("name") for e in entities if e.get("type", '') == n
        ]
        if len(row[entity_type_column]) == 0:
            row[entity_type_column] = ''
    return row


def format_df_named_entity_recognition(
    df: pd.DataFrame,
    api_column_names: NamedTuple,
    entity_types: List,
    column_prefix: AnyStr = "entity_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> pd.DataFrame:
    """
    Format a DataFrame containing API responses for Named Entity Recognition
    """
    logging.info("Formatting API results...")
    df = df.apply(
        func=format_row_named_entity_recognition, axis=1,
        response_column=api_column_names.response, entity_types=entity_types,
        error_handling=error_handling, column_prefix=column_prefix)
    df = move_api_columns_to_end(df, api_column_names)
    logging.info("Formatting API results: Done.")
    return df


def compute_column_description_named_entity_recognition(
    df: pd.DataFrame,
    api_column_names: NamedTuple,
    column_prefix: AnyStr = "entity_api",
) -> Dict:
    """
    Compute dictionary of column descriptions for Named Entity Recognition API
    """
    column_description_dict = {
        v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
        for k, v in api_column_names._asdict().items()}
    for n, m in EntityTypeEnum.__members__.items():
        entity_type_column = generate_unique(
            "entity_type_" + n.lower(), df.keys(), column_prefix)
        column_description_dict[entity_type_column] = \
            "List of '{}' entities recognized by the API".format(str(m.value))
    return column_description_dict


def format_row_text_classification(
    row: Dict,
    response_column: AnyStr,
    num_categories: int = 3,
    column_prefix: AnyStr = "text_classif_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    """
    Format the API response for text classification to:
    - make sure response is valid JSON
    - expand results to multiple JSON columns (one by classification category)
    or put all categories as a list in a single JSON column
    """
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    categories = sorted(
        response.get("categories", []), key=lambda x: x.get("confidence"),
        reverse=True)
    for n in range(num_categories):
        category_column = generate_unique(
            "category_" + str(n+1) + "_name", row.keys(), column_prefix)
        confidence_column = generate_unique(
            "category_" + str(n+1) + "_confidence", row.keys(), column_prefix)
        if len(categories) > n:
            row[category_column] = categories[n].get("name", '')
            row[confidence_column] = categories[n].get("confidence")
        else:
            row[category_column] = ''
            row[confidence_column] = None
    return row


def move_api_columns_to_end(
    df: pd.DataFrame,
    api_column_names: NamedTuple,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Move non-human-readable API columns to the end of the dataframe
    """
    api_column_names_dict = api_column_names._asdict()
    if not verbose:
        api_column_names_dict.pop("error_raw", None)
    api_column_names_list = [v for k, v in api_column_names_dict.items()]
    cols = [
        c for c in list(df.columns.values) if c not in api_column_names_list]
    new_cols = cols + api_column_names_list
    df = df.reindex(columns=new_cols)
    return df
