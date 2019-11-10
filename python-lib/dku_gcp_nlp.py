import json
import logging
from google.cloud import language
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

def get_client(connection_info):
    credentials = _get_credentials(connection_info)
    return language.LanguageServiceClient(credentials=credentials)


def get_sentiment(score, scale):
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
    elif scale == 'continuous':
        return score


def _get_credentials(connection_info):
    if not connection_info.get("credentials"):
        return None
    try:
        credentials = json.loads(connection_info.get("credentials"))
    except Exception as e:
       logging.error(e)
       raise ValueError("Provided credentials are not JSON")
    credentials = service_account.Credentials.from_service_account_info(credentials)
    if hasattr(credentials, 'service_account_email'):
        logging.info("Credentials loaded : %s" % credentials.service_account_email)
    else:
        logging.info("Credentials loaded")
    return credentials
