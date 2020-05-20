# -*- coding: utf-8 -*-
import logging
import json

from google.cloud import language
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.oauth2 import service_account


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DOCUMENT_TYPE = language.enums.Document.Type.PLAIN_TEXT
ENCODING_TYPE = language.enums.EncodingType.UTF8

API_EXCEPTIONS = (GoogleAPICallError, RetryError)


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(gcp_service_account_key=None):
    """
    Get a Google Natural Language API client from the service account key.
    """
    if gcp_service_account_key is None or gcp_service_account_key == "":
        return language.LanguageServiceClient()
    try:
        credentials = json.loads(gcp_service_account_key)
    except Exception as e:
        logging.error(e)
        raise ValueError("GCP service account key is not valid JSON.")
    credentials = service_account.Credentials.from_service_account_info(credentials)
    logging.info("Credentials loaded")
    client = language.LanguageServiceClient(credentials=credentials)
    return client
