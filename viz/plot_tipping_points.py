# Plot Tipping Points
import numpy as np
import requests
from datetime import datetime 
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import xarray as xr
import pandas as pd

os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/viz/')

DATADIR = '/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/data/copernicus_data/'

# Define the pre-industrial period
pre_industrial_start = '1850'
pre_industrial_end = '1900'

experiments = ['historical', 'ssp126', 'ssp245', 'ssp585']

def find_year(temperature):
    year_reached = data_50.year.where(data_50 >= temperature).min().values
    return year_reached

colours = ['gray',' #ff6666', '#FF0000', ' #cc0000']

if __name__ == '__main__':

    data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg*.nc')
    data = data_ds['tas']
    tipping_points = pd.read_csv('../data/McKay_Tipping_Points/tipping_points_for_viz.csv')

    # Calculate the baseline mean temperature for the pre-industrial period
    baseline = data_ds['tas'].sel(year=slice(pre_industrial_start, pre_industrial_end))
    baseline_mean = np.nanmean(baseline)  # use numpy's nanmean

    # Subtract the baseline mean from the temperature values to get anomalies
    anomalies = data_ds['tas'] - baseline_mean

    data_10 = anomalies.quantile(0.1, dim='model')
    data_50 = anomalies.quantile(0.5, dim='model')
    data_90 = anomalies.quantile(0.9, dim='model')

    fig = go.Figure()

    for i in np.arange(len(experiments)):
        fig.add_trace(go.Scatter(x=data_50.year, y=data_50[i,:], mode='lines', 
                                line=dict(color=colours[i]), opacity=0.5, 
                                name=f'{data_50.experiment[i].values}',
                                hoverinfo='none'))
        if i == 0:  # historical 50th quantile
            fig.add_annotation(x=1980, y=data_50[i,:].values[0]+.75, 
                            text=f'{data_50.experiment[i].values}', 
                            showarrow=False, font=dict(color=colours[i]))
        else:  # SSPs
            fig.add_annotation(x=1.08, y=data_50[i,:].values[-1], 
                            text=f'{data_50.experiment[i].values}', 
                            showarrow=False, font=dict(color=colours[i]), xref="paper")


    fig.update_layout(title='Warming is rapidly pushing us toward critical climate tipping points',
                    autosize=False,
                    yaxis_title='Global average temperature anomaly (°C)',
                    xaxis_range=[1960,2100],
                    showlegend=False,
                    height = 600,
                    width = 800
                    )

    # Add a point to the plot for each tipping point
    for index, row in tipping_points.iterrows():
        year = find_year(row['Temperature'])
        fig.add_trace(go.Scatter(x=[year], y=[row['Temperature']], mode='markers', 
                                marker=dict(size=10, symbol='x', color='black'), 
                                name=row['Tipping Point'],
                                hoverinfo='none'))  # remove hover pop-up for the tipping points
        if row['Tipping Point'] in ["Amazon Rainforest Dieback", "West Antarctic Ice Sheet Collapse", "Barents Sea Ice Abrupt Loss"]:
            fig.add_annotation(x=year, y=row['Temperature'], 
                            text=row['Tipping Point'], 
                            showarrow=False, 
                            textangle=30,  # set text angle to slope downwards to the right
                            xshift=-70,  # shift the text to the left
                            yshift=55,  # shift the text up
                            font=dict(color='black', size=10))  # make the font smaller

    print('Writing plots to ../figures/tipping_points.html and ../figures/tipping_points.png')
    pio.write_html(fig, '../figures/tipping_points.html')
    pio.write_image(fig, '../figures/tipping_points.png')

    data_ds.close()