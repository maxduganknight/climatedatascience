import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import plotly.io as pio


os.chdir('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/viz/')
DATADIR = ('../data/global_temperatures')

if __name__ == '__main__':
    df = pd.read_csv(f'{DATADIR}/combined_temps_separate_base.csv')

    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

    # Get the latest date in the data
    latest_date = df['Date'].max()

    fig = go.Figure()

    # Add a line to the plot for each data source
    for column in df.columns:
        if column in ['gistemp', 'berkeley', 'noaa']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[column], name=column))

    fig.update_layout(
        title='Global Mean Temperature', 
        yaxis_title='Temperature Anomaly (C)',
        autosize=False,
        width=1000,  # set the width
        height=800,  # set the height
        xaxis_range=['1950-01-01', latest_date]  # set the x-axis range
    )
    
pio.write_html(fig, '../figures/global_temps.html')
pio.write_image(fig, '../figures/global_temps.png')