import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
from datetime import datetime
import glob
import sys
import os
import json
from contextlib import contextmanager
import shutil

sys.path.append('/Users/max/Deep_Sky')

DATA_DIR = '/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/data'

def get_burnt_area_data(year):
    
    today = datetime.now().strftime('%Y%m%d')
    dataset = "satellite-fire-burned-area"
    output_file = f'{DATA_DIR}/wildfires/2025/burnt_area_{year}_{today}.zip'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    request = {
        "origin": "c3s",
        "sensor": "olci",
        "variable": "grid_variables",
        "version": "1_1",
        "year": str(year),
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "nominal_day": ["01"]
    }

    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(output_file)
        print(f"Success! Data saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error with dataset {dataset}: {e}")



if __name__ == '__main__':    
    
    for year in range(2017, 2023):
        get_burnt_area_data(year)