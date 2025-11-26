<<<<<<< HEAD
#from hmac import new
from email.mime import base
from hmac import new
import re
import argparse
import time
from numpy import number
import requests
import pandas as pd
import json
import sys
import os
from datetime import datetime
sys.path.append('/Users/max/Deep_Sky/')
from creds import CDR_FYI_API_TOKEN
from helpers import send_slack_message, query_cdr_fyi_api

os.chdir('/Users/max/Deep_Sky/GitHub/datascience-platform/scrapers/')

web_hook_url = 'https://hooks.slack.com/services/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'
test_web_hook_url = 'https://hooks.slack.com/services/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'
=======
# from hmac import new
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from email.mime import base
from hmac import new

import pandas as pd
import requests
from numpy import number

sys.path.append("/Users/max/Deep_Sky/")
from creds import CDR_FYI_API_TOKEN
from helpers import query_cdr_fyi_api, send_slack_message

os.chdir("/Users/max/Deep_Sky/GitHub/datascience-platform/scrapers/")

web_hook_url = ""
test_web_hook_url = ""
>>>>>>> fresh-start
cdr_api_token = CDR_FYI_API_TOKEN

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")

<<<<<<< HEAD
# CDR.FYI API Request 

BASE_URL = 'https://api.cdr.fyi/v1'
HEADERS = {
    'Authorization': f'Bearer {cdr_api_token}',
    'x-page': '1',
    'x-limit': '100'
}

def pull_full_table(table):
    """
    Pull full tables once to use as running record of what was already recorded. 
    These tables will be used to compare against API payload to check for updates.
    """
    all_records = pd.DataFrame()
    id_col = str(table[:-1] + '_id')
    i = 1
    while True:
        records = query_cdr_fyi_api(page=i, table=table, BASE_URL=BASE_URL, HEADERS=HEADERS)
        print('Pulled page: {page_number}'.format(page_number=i))
        # As long as there are new records in the page add it to the df and continue. 
=======
# CDR.FYI API Request

BASE_URL = "https://api.cdr.fyi/v1"
HEADERS = {"Authorization": f"Bearer {cdr_api_token}", "x-page": "1", "x-limit": "100"}


def pull_full_table(table):
    """
    Pull full tables once to use as running record of what was already recorded.
    These tables will be used to compare against API payload to check for updates.
    """
    all_records = pd.DataFrame()
    id_col = str(table[:-1] + "_id")
    i = 1
    while True:
        records = query_cdr_fyi_api(
            page=i, table=table, BASE_URL=BASE_URL, HEADERS=HEADERS
        )
        print("Pulled page: {page_number}".format(page_number=i))
        # As long as there are new records in the page add it to the df and continue.
>>>>>>> fresh-start
        # Once there are no more rows, return the whole df.
        if not records.empty:
            records.set_index(id_col, inplace=True)
            # Exclude empty or all-NA columns
<<<<<<< HEAD
            records = records.dropna(how='all', axis=1)
=======
            records = records.dropna(how="all", axis=1)
>>>>>>> fresh-start
            all_records = pd.concat([all_records, records])
            i += 1
        else:
            return all_records
<<<<<<< HEAD
    
def update_checker(table_name, output_file_suffix):
    """
    Checks for new rows in the specified table from the CDR_FYI API, 
    updates the local CSV file if new rows are found, and returns the new rows.
    """
    new_rows = False
    old_table_path = '../data/CDR_FYI_Data/cdr_fyi_{table}{suffix}'.format(table=table_name, suffix = output_file_suffix)
=======


def update_checker(table_name, output_file_suffix):
    """
    Checks for new rows in the specified table from the CDR_FYI API,
    updates the local CSV file if new rows are found, and returns the new rows.
    """
    new_rows = False
    old_table_path = "../data/CDR_FYI_Data/cdr_fyi_{table}{suffix}".format(
        table=table_name, suffix=output_file_suffix
    )
>>>>>>> fresh-start
    old_table = pd.read_csv(old_table_path, index_col=0)
    old_ids = old_table.index.tolist()
    print("Retrieving {table} data from CDR.FYI API.".format(table=table_name))
    pulled_table = pull_full_table(table_name)
    new_rows = pulled_table[~pulled_table.index.isin(old_ids)]

    if new_rows.shape[0] > 0:
<<<<<<< HEAD
        new_rows = new_rows.dropna(how='all', axis=1)
        updated_table = pd.concat([old_table, new_rows])
        updated_table_path = '../data/CDR_FYI_Data/cdr_fyi_{table}{suffix}'.format(table=table_name, suffix=output_file_suffix)
        updated_table.to_csv(updated_table_path)
        print('Updates found and added to {updated_table}\n'.format(updated_table=updated_table_path))
=======
        new_rows = new_rows.dropna(how="all", axis=1)
        updated_table = pd.concat([old_table, new_rows])
        updated_table_path = "../data/CDR_FYI_Data/cdr_fyi_{table}{suffix}".format(
            table=table_name, suffix=output_file_suffix
        )
        updated_table.to_csv(updated_table_path)
        print(
            "Updates found and added to {updated_table}\n".format(
                updated_table=updated_table_path
            )
        )
>>>>>>> fresh-start
    else:
        updated_table = old_table
        print("No updates found.")
    return new_rows

<<<<<<< HEAD
=======

>>>>>>> fresh-start
def new_orders_slack_message_builder(new_orders, output_file_suffix):
    """
    Build message to send on slack and return string of message.
    """
    orders_message = "{number_of_new_orders} new orders since last check.\nSee details below and go to the Data Explorer tab at CDR.FYI for more info.\n\n".format(
        number_of_new_orders=new_orders.shape[0]
<<<<<<< HEAD
        )
    # read in suppliers and purchasers
    all_suppliers = pd.read_csv('../Data/CDR_FYI_Data/cdr_fyi_suppliers{suffix}'.format(suffix = output_file_suffix))
    all_purchasers = pd.read_csv('../Data/CDR_FYI_Data/cdr_fyi_purchasers{suffix}'.format(suffix = output_file_suffix))
=======
    )
    # read in suppliers and purchasers
    all_suppliers = pd.read_csv(
        "../Data/CDR_FYI_Data/cdr_fyi_suppliers{suffix}".format(
            suffix=output_file_suffix
        )
    )
    all_purchasers = pd.read_csv(
        "../Data/CDR_FYI_Data/cdr_fyi_purchasers{suffix}".format(
            suffix=output_file_suffix
        )
    )
>>>>>>> fresh-start
    order_number = 0
    order_dict = {}
    for index, order in new_orders.iterrows():
        order_number += 1
        # find the names of the suppliers and purchasers involved in new orders
<<<<<<< HEAD
        supplier = all_suppliers[all_suppliers['supplier_id'] == order['supplier_id']]['name']
        purchaser = all_purchasers[all_purchasers['purchaser_id'] == order['purchaser_id']]['name']
        if not supplier.empty:
            supplier_name = supplier.iloc[0]
        else:
            supplier_name = 'an unspecified supplier'
        if not purchaser.empty:
            purchaser_name = purchaser.iloc[0]
        else:
            purchaser_name = 'An unspecified purchaser'
        if order['tons_purchased']:
            tons_purchased = order['tons_purchased']
        else:
            tons_purchased = 'unspecified number of'
        if order['method']:
            removal_method = order['method']
        else:
            removal_method = 'unspecified'
        if order['announcement_date']:
            announcement_date = pd.to_datetime(order['announcement_date']).strftime('%B %d, %Y')
        else:
            announcement_date = 'unspecified date'
=======
        supplier = all_suppliers[all_suppliers["supplier_id"] == order["supplier_id"]][
            "name"
        ]
        purchaser = all_purchasers[
            all_purchasers["purchaser_id"] == order["purchaser_id"]
        ]["name"]
        if not supplier.empty:
            supplier_name = supplier.iloc[0]
        else:
            supplier_name = "an unspecified supplier"
        if not purchaser.empty:
            purchaser_name = purchaser.iloc[0]
        else:
            purchaser_name = "An unspecified purchaser"
        if order["tons_purchased"]:
            tons_purchased = order["tons_purchased"]
        else:
            tons_purchased = "unspecified number of"
        if order["method"]:
            removal_method = order["method"]
        else:
            removal_method = "unspecified"
        if order["announcement_date"]:
            announcement_date = pd.to_datetime(order["announcement_date"]).strftime(
                "%B %d, %Y"
            )
        else:
            announcement_date = "unspecified date"
>>>>>>> fresh-start
        orders_message += "{number}. {purchaser} purchased {tons} tons from {supplier} with {method} method on {date}.\n".format(
            number=order_number,
            supplier=supplier_name,
            tons=tons_purchased,
            purchaser=purchaser_name,
            method=removal_method,
<<<<<<< HEAD
            date=announcement_date
            )
    return orders_message

=======
            date=announcement_date,
        )
    return orders_message


>>>>>>> fresh-start
def new_participant_slack_message_builder(new_rows, table_name):
    """
    Build message to send on slack and return string of message. Used for both suppliers and purchasers.
    """
    message = "{number_of_new_rows} new {table}s since last check.\nSee details below and go to the Data Explorer tab at CDR.FYI for more info.\n".format(
<<<<<<< HEAD
        number_of_new_rows=new_rows.shape[0],
        table=table_name
        )
=======
        number_of_new_rows=new_rows.shape[0], table=table_name
    )
>>>>>>> fresh-start
    row_number = 0
    for index, org in new_rows.iterrows():
        row_number += 1
        message += "{number}. {org} added as new {table}. ({website})\n".format(
<<<<<<< HEAD
            number=row_number,
            org=org['name'],
            table = table_name,
            website = org['website']
            )
    return message

if __name__== "__main__":

    # Should only need to run below commented out lines once to initially 
=======
            number=row_number, org=org["name"], table=table_name, website=org["website"]
        )
    return message


if __name__ == "__main__":
    # Should only need to run below commented out lines once to initially
>>>>>>> fresh-start
    # download all available data for each of orders, suppliers, and purchasers

    # table = 'orders'
    # df = pull_full_table(table)
    # df.to_csv('../data/CDR_FYI_Data/cdr_fyi_{table}_test.csv'.format(table=table))

    # Parse command line arguments
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode')
    parser.add_argument('-s', '--slack', action='store_true', help='Send a Slack message')
    args = parser.parse_args()
    
    # Use the test webhook URL and output file if the test argument is provided
    if args.test:
        print('~TEST RUN~')
        web_hook_url = test_web_hook_url
        output_file_suffix = '_test.csv'
    else:
        output_file_suffix = '.csv'

    print('\nScript called at {now}'.format(now=formatted_date_time))
    new_suppliers = update_checker('suppliers', output_file_suffix)
    new_purchasers = update_checker('purchasers', output_file_suffix)
    new_orders = update_checker('orders', output_file_suffix)
    new_marketplaces = update_checker('marketplaces', output_file_suffix)
    if args.slack:
        if new_orders.shape[0] > 0:
            print('Sending new orders slack message.')
            message = new_orders_slack_message_builder(new_orders, output_file_suffix)
            send_slack_message(web_hook_url, message)
        
        if new_suppliers.shape[0] > 0:
            print('Sending new suppliers slack message.')
            message = new_participant_slack_message_builder(new_suppliers, 'supplier')
            send_slack_message(web_hook_url, message)

        if new_purchasers.shape[0] > 0:
            print('Sending new purchasers slack message.')
            message = new_participant_slack_message_builder(new_purchasers, 'purchaser')
            send_slack_message(web_hook_url, message)

        if new_marketplaces.shape[0] > 0:
            print('Sending new marketplaces slack message.')
            message = new_participant_slack_message_builder(new_marketplaces, 'marketplace')
=======
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "-s", "--slack", action="store_true", help="Send a Slack message"
    )
    args = parser.parse_args()

    # Use the test webhook URL and output file if the test argument is provided
    if args.test:
        print("~TEST RUN~")
        web_hook_url = test_web_hook_url
        output_file_suffix = "_test.csv"
    else:
        output_file_suffix = ".csv"

    print("\nScript called at {now}".format(now=formatted_date_time))
    new_suppliers = update_checker("suppliers", output_file_suffix)
    new_purchasers = update_checker("purchasers", output_file_suffix)
    new_orders = update_checker("orders", output_file_suffix)
    new_marketplaces = update_checker("marketplaces", output_file_suffix)
    if args.slack:
        if new_orders.shape[0] > 0:
            print("Sending new orders slack message.")
            message = new_orders_slack_message_builder(new_orders, output_file_suffix)
            send_slack_message(web_hook_url, message)

        if new_suppliers.shape[0] > 0:
            print("Sending new suppliers slack message.")
            message = new_participant_slack_message_builder(new_suppliers, "supplier")
            send_slack_message(web_hook_url, message)

        if new_purchasers.shape[0] > 0:
            print("Sending new purchasers slack message.")
            message = new_participant_slack_message_builder(new_purchasers, "purchaser")
            send_slack_message(web_hook_url, message)

        if new_marketplaces.shape[0] > 0:
            print("Sending new marketplaces slack message.")
            message = new_participant_slack_message_builder(
                new_marketplaces, "marketplace"
            )
>>>>>>> fresh-start
            send_slack_message(web_hook_url, message)

    else:
        pass
<<<<<<< HEAD





=======
>>>>>>> fresh-start
