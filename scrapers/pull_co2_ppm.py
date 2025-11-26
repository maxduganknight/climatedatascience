import requests
from datetime import datetime 
import pandas as pd
from helpers import download_file_from_url
import os

#Set to your working directory
os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/scrapers/')

current_date = datetime.now().strftime('%Y-%m-%d')

# Download ppm data from noaa website
url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv'
local_filename = f'../data/CO2_PPM/co2_ppm_{current_date}.csv'
download_file_from_url(url, local_filename)

# atmospheric growth rate
growth_url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_gr_mlo.csv'
growth_local_filename = f'../data/CO2_PPM/co2_growth_rate_{current_date}.csv'
download_file_from_url(growth_url, growth_local_filename)
