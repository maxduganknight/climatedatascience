import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 

import glob
import sys
import os

# creds
sys.path.append('/Users/max/Deep_Sky')
from creds import CDS_UID, CDS_API_KEY_2

c = cdsapi.Client()

def retrieve_ERA5_daily(variables,
                  folder,
                  target_months=[
                      '01',
                      '02',
                      '03',
                      '04',
                      '05',
                      '06',
                      '07',
                      '08',
                      '09',
                      '10',
                      '11',
                      '12',
                  ],
                  area=[90, -180, -90, 180],
                  years=np.arange(1979, 2025)):
    """Retrieves the full ERA5 dataset from CDS (years 1979-2020).
        
        Parameters
        ----------
        variables : The variables to be downloaded, str. Can also be one variable.
        folder : The path to the folder where to store the data. 
        grid : The grid spacing, or spatial resolution of the data. 
            Defaults to 1x1 degrees to match SEAS5.
            If a higher resolution is wanted, use [0.25, 0.25].
        target_months : The month(s) of interest.
            For example, for JJA, use [6,7,8]. 
            Defaults to all months. 
        area : The domain to download the data over, [North, West, South, East,].
            For example, to dowload longitude 30,70 and latitude -10, 120, use [70, -11, 30, 120,].
            Default is the full extent [90, -180, -90, 180].
        years : Defaults to the period 1979-2020.

        Returns
        -------
        Saves the files in the specified folder.
    """
    for j in range(len(years)):
        year = years[j]
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'format': 'netcdf',
                'product_type': 'reanalysis',
                'variable': variables,
                'area': area,         
                'year': str(year),
                'month': target_months,
                'day': [
                    '1', '2', '3', '4', '5'
                    '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15',
                    '16', '17', '18', '19', '20',
                    '21', '22', '23', '24', '25',
                    '26', '27', '28', '29', '30', '31'
                ],
                    'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
            },
            folder + 'ERA5_' + str(year) + '.nc')
        
if __name__ == '__main__':
    
    area_of_study = [36.5, -98, 24.5, -75]

    retrieve_ERA5_daily(variables='total_precipitation',
                       target_months=[6,7,8,9,10],
                       years = np.arange(1980,2025),
                       area=area_of_study,
                       folder='rainfall_data/rainfall_raw/')