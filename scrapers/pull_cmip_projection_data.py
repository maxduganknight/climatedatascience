# General libs for file paths, data extraction, etc
from glob import glob
from pathlib import Path
from os.path import basename
import zipfile # To extract zipfiles
import urllib3 
urllib3.disable_warnings() # Disable warnings for data download via API

# CDS API
import cdsapi

# Libraries for working with multi-dimensional arrays
import numpy as np
import xarray as xr
import pandas as pd

# creds
import sys
import os
sys.path.append('/Users/max/Deep_Sky/')
from creds import CDS_UID, CDS_API_key

URL = 'https://cds.climate.copernicus.eu/api/v2'
os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/scrapers/')
DATADIR = '../data/copernicus_data/'

experiments = ['historical', 'ssp1_1_9', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

models = [
    #'hadgem3_gc31_ll', 
          #'inm_cm5_0', 
        #   'inm_cm4_8', 'ipsl_cm6a_lr', 
        #   'miroc_es2l', 'mpi_esm1_2_lr', 'ukesm1_0_ll',
          #'giss_e2_1_g', 
          #'ec-earth3-veg',
          'gfdl-esm4',
          'ec-earth3',
          'canesm5'
          ]

# Define the pre-industrial period
pre_industrial_start = '1850'
pre_industrial_end = '1900'

c = cdsapi.Client(url=URL, key="{id}:{key}".format(id = CDS_UID, key = CDS_API_key))

def geog_agg(fn):
    # MDK getting this error when I try to open these files in plot_tipping_points.py
    # ValueError: Resulting object does not have monotonic global indexes along dimension year
    print(fn)
    ds = xr.open_dataset(f'{DATADIR}{fn}')
    exp = ds.attrs['experiment_id']
    mod = ds.attrs['source_id']
    da = ds['tas']
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    da_agg = da_weighted.mean(['lat', 'lon'])
    da_yr = da_agg.groupby('time.year').mean()
    da_yr = da_yr.sortby('year')
    da_yr = da_yr - 273.15
    da_yr = da_yr.assign_coords(model=mod)
    da_yr = da_yr.expand_dims('model')
    da_yr = da_yr.assign_coords(experiment=exp)
    da_yr = da_yr.expand_dims('experiment')
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')


if __name__ == '__main__':

    # DOWNLOAD DATA FOR HISTORICAL PERIOD

    # for j in models:
    #     c.retrieve(
    #         'projections-cmip6',
    #         {
    #             'format': 'zip',
    #             'temporal_resolution': 'monthly',
    #             'experiment': 'historical',
    #             'level': 'single_levels',
    #             'variable': 'near_surface_air_temperature',
    #             'model': f'{j}',
    #             'date': '1850-01-01/2014-12-31',
    #         },
    #         f'{DATADIR}cmip6_monthly_1850-2014_historical_{j}.zip')
        
    # DOWNLOAD DATA FOR FUTURE SCENARIOS

    # for i in experiments[1:]:
    #     for j in models:
    #         c.retrieve(
    #             'projections-cmip6',
    #             {
    #                 'format': 'zip',
    #                 'temporal_resolution': 'monthly',
    #                 'experiment': f'{i}',
    #                 'level': 'single_levels',
    #                 'variable': 'near_surface_air_temperature',
    #                 'model': f'{j}',
    #                 'date': '2015-01-01/2100-12-31',
    #             },
    #             f'{DATADIR}cmip6_monthly_2015-2100_{i}_{j}.zip')

    experiment = 'ssp1_1_9'
    model = 'ec_earth3_veg'

    c.retrieve(
        'projections-cmip6',
        {
            'format': 'zip',
            'temporal_resolution': 'monthly',
            'experiment': experiment,
            'level': 'single_levels',
            'variable': 'near_surface_air_temperature',
            'model': model,
            'date': '2015-01-01/2100-12-31',
        },
        f'{DATADIR}cmip6_monthly_2015-2100_{experiment}_{model}.zip')
            

    cmip6_zip_paths = glob(f'{DATADIR}*.zip')
    for j in cmip6_zip_paths:
        with zipfile.ZipFile(j, 'r') as zip_ref:
            zip_ref.extractall(f'{DATADIR}')

    cmip6_nc = list()
    cmip6_nc_rel = glob(f'{DATADIR}tas*.nc')
    for i in cmip6_nc_rel:
        cmip6_nc.append(os.path.basename(i))

    # ds = xr.open_dataset(f'{DATADIR}{cmip6_nc[0]}')

    # exp = ds.attrs['experiment_id']
    # mod = ds.attrs['source_id']

    # da = ds['tas']

    # weights = np.cos(np.deg2rad(da.lat))
    # weights.name = "weights"
    # da_weighted = da.weighted(weights)

    # da_agg = da_weighted.mean(['lat', 'lon'])
    # da_yr = da_agg.groupby('time.year').mean()
    # da_yr = da_yr - 273.15
    # da_yr = da_yr.assign_coords(model=mod)
    # da_yr = da_yr.expand_dims('model')
    # da_yr = da_yr.assign_coords(experiment=exp)
    # da_yr = da_yr.expand_dims('experiment')

    # for i in cmip6_nc:
    #     try:
    #         geog_agg(i)
    #     except: print(f'{i} failed')
