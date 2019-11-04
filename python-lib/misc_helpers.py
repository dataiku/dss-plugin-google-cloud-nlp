import json
import os
import logging
from google.oauth2 import service_account

def get_credentials(connection_info):
    print("######"+ json.dumps(connection_info))
    if connection_info.get("credentials") is None:
        return None
    try:
        obj = json.loads(connection_info.get("credentials"))
    except Exception as e:
        logging.error(e)
        raise ValueError("Provided credentials are not JSON")
    credentials = service_account.Credentials.from_service_account_info(obj)
    _log_get_credentials(credentials)
    return credentials

def _log_get_credentials(credentials):
    if hasattr(credentials, 'service_account_email'):
        logging.info("Credentials loaded : %s" % credentials.service_account_email)
    else:
        logging.info("Credentials loaded")
