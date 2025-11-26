# Plot CO2 PPM data from the Mauna Loa Observatory

import numpy as np
import requests
from datetime import datetime 
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/viz/')

infile = '../data/CO2_PPM/co2_ppm.csv'
outdir = '../figures/'

if os.path.exists(infile):
    df = pd.read_csv(infile, skiprows=40)
else:
    print(f"The file {infile} does not exist.")
    print("Available files:")
    for filename in os.listdir('../data/CO2_PPM/'):
        if filename.endswith('.csv'):
            print(filename)

# Ensure your DataFrame is sorted by date
df = df.sort_values('decimal date')

# Calculate the percentage change compared to the same month of the previous year
df['average_abs_change'] = round(df['average'] - df['average'].shift(12), 3)

latest_co2_ppm = df.iloc[-1]
today_date = datetime.today().date()

# Create a plot
fig = px.line(
    df, x='decimal date', y='average', 
    labels={'decimal date': 'Year', 'average': 'Monthly Average CO2 (ppm)'}, 
    title='Atmospheric CO2 Concentration, {date}'.format(latest_value = latest_co2_ppm['average'], date = today_date),
    hover_data={'average_abs_change': True},  # Add average_abs_change to hover_data
    hover_name='year',  # Use 'year' for hover name
    custom_data=['average_abs_change']  # Add average_abs_change to custom_data
)

fig.add_trace(go.Scatter(x=df['decimal date'], y=df['deseasonalized'], mode='lines', name='Deseasonalized', hoverinfo='none', showlegend=False))


# Update hovertemplate to include average_abs_change with custom label
fig.update_traces(hovertemplate='Year: %{hovertext}<br>Monthly Average CO2 (ppm): %{y}<br>Change From This Month Last Year: +%{customdata[0]} ppm')

fig.update_layout(xaxis_title=None)

fig.add_shape(type="line", x0=df['decimal date'].min(), x1=df['decimal date'].max(), 
              y0=330, y1=330, line=dict(color="Red", width=2, dash="dash"))

fig.add_annotation(x=1970, y=330, text="Safe level", showarrow=False, yshift=10)

# Add an annotation for the latest data point
fig.add_annotation(
    x=latest_co2_ppm['decimal date'], 
    y=latest_co2_ppm['average'], 
    text=f"Latest: {latest_co2_ppm['average']} ppm",
    showarrow=True,
    arrowhead=1,
    ax=-100,  
    ay=0)

print(f"Latest CO2 PPM: {latest_co2_ppm['average']} ppm")
print('Writing plots to {filename}'.format(filename = outdir))
pio.write_html(fig, '{path}/CO2_PPM_plot.html'.format(path = outdir))
pio.write_image(fig, '{path}/CO2_PPM_plot.png'.format(path = outdir))