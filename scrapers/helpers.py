import numpy as np
import requests
from datetime import datetime 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def download_file_from_url(url, filename):
    response = requests.get(url)
    response.raise_for_status()  
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f'Downloaded {filename}')

def send_slack_message(webhook_url, message):
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Request to Slack returned an error {response.status_code}, the response is:\n{response.text}")

def query_cdr_fyi_api(table, BASE_URL, HEADERS, page=1, limit=100, entity_filter_type=None, entity_filter_id=None, ):
    """
    Retrieve orders from the API and return as a DataFrame.
    """
    url = "{BASE_URL}/{DATA_TYPE}".format(BASE_URL = BASE_URL, DATA_TYPE = table)
    request_headers = HEADERS.copy()
    request_headers.update({
        "x-page": str(page),
        "x-limit": str(limit)
    })
    params = {}
    if entity_filter_type is not None:
        params["entityFilterType"] = entity_filter_type
    if entity_filter_id is not None:
        params["entityFilterId"] = entity_filter_id
    response = requests.get(url, headers=request_headers, params=params)
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON. Response text:")
        print(response.text)
        return None, False
    df = pd.json_normalize(data[table])
    return df