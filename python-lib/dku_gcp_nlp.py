import json
import logging
from google.cloud import language
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

ALL_ENTITY_TYPES = ['UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION', 'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER', 'PHONE_NUMBER', 'ADDRESS', 'DATE', 'NUMBER', 'PRICE']

def get_client(connection_info):
    credentials = _get_credentials(connection_info)
    return(language.LanguageServiceClient(credentials = credentials))

def _distinct(l):
    return list(dict.fromkeys(l))

def format_entities_results(raw_results, scale=None):
    result = json.loads(MessageToJson(raw_results))
    output_row = dict()
    output_row['entities'] = result.get('entities')
    output_row["raw_results"] = result
    for t in ALL_ENTITY_TYPES:
        output_row[t] = _distinct([e["name"] for e in output_row["entities"] if e["type"] == t])
    return(output_row)

def format_classification_results(raw_results):
    result = json.loads(MessageToJson(raw_results))
    output_row = dict()
    output_row['categories'] = [c['name'] for c in result.get('categories', [])]
    #if remove_prefix:
    #    output_row['categories'] = [c.split('/')[-1] for c in output_row['categories']]
    output_row["raw_results"] = result
    return(output_row)

def format_sentiment_results(raw_results, scale=None):
    result = json.loads(MessageToJson(raw_results))
    output_row = dict()
    score = result.get("documentSentiment", {}).get("score")
    if score is not None:
        output_row['predicted_sentiment'] = format_sentiment(score, scale)
        output_row['detected_language'] = result.get('language')
        output_row["raw_results"] = result
    else:
        logging.warn("API did not return sentiment")
        output_row['predicted_sentiment'] = None
        output_row['detected_language'] = None
        output_row["raw_results"] = None
    return(output_row)

def format_sentiment(score, scale):
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
        return(round((score+1)/2, 2))
    else:
        return(round(score, 2))

def _get_credentials(connection_info):
    if not connection_info.get("credentials"):
        return(None)
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
    return(credentials)
