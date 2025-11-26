def season_mean(ds, years, calendar='standard'):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.year').sum().values, np.ones(years))

    # Calculate the weighted average
    return (ds * weights).groupby('time.year').sum(dim='time', min_count = 1)

def count_extreme_fwi_days(location_data, all_data):
    # Calculate the 90th, 95th, and 99th percentiles between 1994 and 2014
    ds_1983_2023 = all_data.sel(time=slice('1983-01-01', '2022-12-31')).chunk({'time': -1})  # rechunking here
    p90 = ds_1983_2023['fwinx'].quantile(0.9).values
    p95 = ds_1983_2023['fwinx'].quantile(0.95).values
    p99 = ds_1983_2023['fwinx'].quantile(0.99).values
    p995 = ds_1983_2023['fwinx'].quantile(0.995).values

    # Initialize a DataFrame to store the counts
    extreme_fwi_days_df = pd.DataFrame(columns=['Year', 'p90_count', 'p95_count', 'p99_count', 'p995_count'])

    # For each year, count the number of days above each percentile in the specific location
    for year in pd.to_datetime(location_data['time'].values).year.unique():
        ds_year = location_data.sel(time=str(year))
        p90_count = (ds_year['fwinx'] > p90).sum().values
        p95_count = (ds_year['fwinx'] > p95).sum().values
        p99_count = (ds_year['fwinx'] > p99).sum().values
        p995_count = (ds_year['fwinx'] > p995).sum().values

        new_row = pd.DataFrame({'Year': [year], 'p90_count': [p90_count], 'p95_count': [p95_count], 'p99_count': [p99_count], 'p995_count': [p995_count]})
        extreme_fwi_days_df = pd.concat([extreme_fwi_days_df, new_row], ignore_index=True)
    return extreme_fwi_days_df

def count_extreme_fwi_days_by_month(ds):
    # Initialize a DataFrame to store the counts
    extreme_fwi_days_df = pd.DataFrame(index=pd.to_datetime(ds['time'].values).year.unique(), 
                                       columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Calculate the 95th percentile for the entire period between 1994 and 2004
    ds_1983_2023 = ds.sel(time=slice('1982-01-01', '2022-12-31')).chunk({'time': -1})
    p95_total = ds_1983_2023.quantile(0.995)['fwinx'].values

    # For each year and each month, count the number of days above the 95th percentile
    for year in extreme_fwi_days_df.index:
        for month in range(1, 13):
            ds_year_month = ds.sel(time=(ds['time.year'] == year) & (ds['time.month'] == month))
            count = (ds_year_month['fwinx'] > p95_total).sum().values
            extreme_fwi_days_df.loc[year, extreme_fwi_days_df.columns[month-1]] = count

    # Reset the index to make 'Year' a column
    extreme_fwi_days_df.reset_index(inplace=True)
    extreme_fwi_days_df.rename(columns={'index': 'Year'}, inplace=True)

    return extreme_fwi_days_df    

def get_shape_us(shapefile, state_abbreviation):
    shapefile = shapefile.to_crs("EPSG:4326")
    state_shape = shapefile[shapefile.STUSPS == state_abbreviation].geometry
    return state_shape

def get_shape_ca(shapefile, province_name):
    shapefile = shapefile.to_crs("EPSG:4326")
    province_shape = shapefile[shapefile.PRENAME == province_name].geometry
    return province_shape

def mask_xr_dataset(xr_data, shape):
    xr_data = xr_data.assign_coords(longitude=(((xr_data.longitude + 180) % 360) - 180)).sortby('longitude')
    xr_data.rio.write_crs("EPSG:4326", inplace=True)
    xr_data_masked = xr_data.rio.clip(shape.geometry.apply(mapping), shape.crs)
    return xr_data_masked

def aggregate_annually(xr_data, vars):
    xr_data_yearly = season_mean(xr_data, len(pd.to_datetime(xr_data.time.values).year.unique()))
    area_weights = np.cos(np.deg2rad(xr_data.latitude))
    events = (
        xr_data_yearly[vars].       
        weighted(area_weights).
        mean(['longitude', 'latitude'])
    )
    return events

def store_xr_events_data(events, location):
    df_output = events.to_dataframe().reset_index()
    df_output = df_output.rename(columns={'fwinx': 'mean_fire_weather_index', 'year': 'Year'}, )  # rename 'fwinx' to 'mean_fire_weather_index'
    df_output = df_output[['Year', 'mean_fire_weather_index']]  # keep only 'year' and 'fwinx' columns
    df_output.to_csv(f'../../data/UNSEEN/wildfire/cems/preprocessed/canada/ERA5_mean_fwi_{location}.csv', index=False)