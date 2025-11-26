import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 
from datetime import datetime
import glob
import sys
import os
from contextlib import contextmanager
import shutil

# creds
sys.path.append('/Users/max/Deep_Sky')
# from creds import CAMS_UID, CAMS_API_KEY

DATADIR = '/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/wildfires_2025/emissions'
output_file = os.path.join(DATADIR, 'cems_fire_emissions_2023-2025.nc')

dataset = "cams-global-fire-emissions-gfas"
request = {
    "variable": ["wildfire_flux_of_carbon_dioxide"],
    "date": ["2023-01-01/2025-04-29"],
    "data_format": "netcdf"
}

client = cdsapi.Client()

client.retrieve(dataset, request).download(output_file)