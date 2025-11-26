import pandas as pd
import os, io, requests
from datetime import datetime
from functools import reduce

today = datetime.today()

#Set to your working directory
os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/scrapers/')
DATADIR = '../data/global_temperatures/'

#File URLs

gistemp_file = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt' #v4
#gistemp_file = 'https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.csv' #v3
noaa_file = 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/1880-2024/data.csv'
#noaa_file = 'https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/p12/12/1880-2050.csv'
hadley_file = 'https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.monthly_ns_avg.txt'
berkeley_file = 'https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt'
#berkeley_file = 'http://berkeleyearth.lbl.gov/auto/Global/Land_and_Ocean_complete.txt'
cowtan_way_file = 'http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.txt'


cur_month = str(today.month).rjust(2, '0')
cur_year = str(today.year)
copernicus_file = 'https://climate.copernicus.eu/sites/default/files/'+cur_year+'-'+cur_month+'/ts_1month_anomaly_Global_ei_2T_'

#Common baseline period (inclusive)
to_rebaseline = False
start_year = 1961
end_year = 1990


def import_gistemp(filename):
    '''
    Import NASA's GISTEMP, reformatting it to long
    '''
    print('Importing GISTEMP')
    urlData = requests.get(gistemp_file).content
    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')), skiprows=7, sep=r'\s+')
    df.drop(['J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON', 'Year.1'], axis=1, inplace=True)
    for num in range(1, 13):
        df.columns.values[num] = 'month_'+str(num)
    df = df[~df['Year'].isin(['Year', 'Divide', 'Multiply', 'Example', 'change'])]
    df_long = pd.wide_to_long(df.reset_index(), ['month_'], i='Year', j='month').reset_index()
    df_long.columns = ['year', 'month', 'index', 'gistemp']
    df_long.drop(columns='index', inplace=True)
    df_long = df_long.apply(pd.to_numeric, errors='coerce')
    df_long.sort_values(by=['year', 'month'], inplace=True)
    df_long['gistemp'] = df_long['gistemp'] / 100.
    df_long.reset_index(inplace=True, drop=True)
    return df_long


def import_noaa(filename):
    '''
    Import NOAA's GlobalTemp
    '''
    print('Importing NOAA')
    df = pd.read_csv(filename, skiprows=5, names=['date', 'noaa'])
    df['year'] = df['date'].astype(str).str[:4]
    df['month'] = df['date'].astype(str).str[4:6]
    df.drop(['date'], axis=1, inplace=True)
    df = df[['year', 'month', 'noaa']].apply(pd.to_numeric)
    print(df.head())
    return df


def import_hadley(filename):
    '''
    Import Hadley's HadCRUT4
    '''
    print('Importing Hadley')
    urlData = requests.get(hadley_file).content
    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')), header=None, sep=r'\s+', usecols=[0,1], names=['date', 'hadcrut4'])
    df['year'] = df['date'].astype(str).str[:4]
    df['month'] = df['date'].astype(str).str[5:7]
    df = df[['year', 'month', 'hadcrut4']].apply(pd.to_numeric)
    return df


def import_berkeley(filename):
    '''
    Import Berkeley Earth. Keep only the operational air-over-sea-ice varient from the file.
    '''
    print('Importing Berkeley Earth')
    df = pd.read_csv(berkeley_file, skiprows=85, sep=r'\s+', header=None, usecols=[0,1,2], names=['year', 'month', 'berkeley'])
    end_pos = df.index[df['month'] == 'Global'].tolist()[0]
    return df[0:end_pos].apply(pd.to_numeric)


def import_cowtan_way(filename):
    '''
    Import Cowtan and Way's temperature record.
    '''
    print('Importing Cowtan and Way')
    df = pd.read_csv(filename, sep=r'\s+',header=None, usecols=[0,1], names=['date', 'cowtan_way'])
    df['year'] = df['date'].astype(str).str[:4].astype(int)
    df['month'] = ((df['date'] - df['year']) * 12 + 1).astype(int)
    df = df[['year', 'month', 'cowtan_way']].apply(pd.to_numeric)
    return df


def import_copernicus():
    '''
    Import Copernicus/ECMWF reanalysis surface temperature record.
    '''  
    print('Importing Copernicus')
    copernicus_filename = find_copernicus_file()
    df = pd.read_csv(copernicus_filename, header=1)
    df.columns = ['date', 'copernicus', 'copernicus_europe']
    df['year'] = df['date'].astype(str).str[:4]
    df['month'] = df['date'].astype(str).str[4:6]
    df = df[['year', 'month', 'copernicus']].apply(pd.to_numeric)
    return df


def find_copernicus_file():
    '''
    Identify the correct Copernicus file to use based on the current month.
    Should prevent problems on the first day of the month when the record
    has yet to be updated.
    '''
    try:
        month = str(today.month).rjust(2, '0')
        copernicus_filename = copernicus_file+str(today.year)+month+'.csv'
        pd.read_csv(copernicus_filename, header=1)
    except:
        month = str(today.month - 1).rjust(2, '0')
        copernicus_filename = copernicus_file+str(today.year)+month+'.csv'
        pd.read_csv(copernicus_filename, header=1)
    return copernicus_filename


def combined_global_temps(start_year, end_year, to_rebaseline=True):
    '''
    Merge all the files together, rebaselining them all to a common period.
    '''
    if to_rebaseline == True:
        hadley = rebaseline(import_hadley(hadley_file), start_year, end_year)
        gistemp = rebaseline(import_gistemp(gistemp_file), start_year, end_year)
        noaa = rebaseline(import_noaa(noaa_file), start_year, end_year)
        berkeley = rebaseline(import_berkeley(berkeley_file), start_year, end_year)
        cowtan_and_way = rebaseline(import_cowtan_way(cowtan_way_file), start_year, end_year)
        #copernicus = rebaseline(import_copernicus(), start_year, end_year)
        #dfs = [hadley, gistemp, noaa, berkeley, cowtan_and_way, copernicus]
        dfs = [hadley, gistemp, noaa, berkeley, cowtan_and_way]
        df_final = reduce(lambda left,right: pd.merge(left,right,on=['year', 'month'], how='outer'), dfs)
    else:
        hadley = import_hadley(hadley_file)
        gistemp = import_gistemp(gistemp_file)
        noaa = import_noaa(noaa_file)
        berkeley = import_berkeley(berkeley_file)
        cowtan_and_way = import_cowtan_way(cowtan_way_file)
        #copernicus = import_copernicus()
        #dfs = [hadley, gistemp, noaa, berkeley, cowtan_and_way, copernicus]
        dfs = [hadley, gistemp, noaa, berkeley, cowtan_and_way]
        df_final = reduce(lambda left,right: pd.merge(left,right,on=['year', 'month'], how='outer'), dfs)
    return df_final.round(3)


def rebaseline(temps, start_year, end_year):
    '''
    Rebaseline data by subtracting the mean value between the start and
    end years from the series.
    '''
    mean = temps[
        temps['year'].between(start_year, end_year, inclusive=True)
    ].iloc[:, 2].mean()
    temps.iloc[:, 2] -= mean
    return temps

if __name__ == "__main__":
    combined_temps = combined_global_temps(start_year, end_year, to_rebaseline)
    if to_rebaseline == True:
        combined_temps.to_csv(f'{DATADIR}combined_temps_base_'+str(start_year)+'_to_'+str(end_year)+'.csv')
    else:
        combined_temps.to_csv(f'{DATADIR}combined_temps_separate_base.csv')