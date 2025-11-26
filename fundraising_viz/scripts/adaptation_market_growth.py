"""
Adaptation Market Growth Visualization

This script visualizes the growth of climate adaptation markets including:
- Size of adaptation solutions (market cap and revenue)
- Adaptation-related green bond issuance by economy type
- Catastrophe bonds and ILS cumulative issuance over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to path for utils import
sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def load_lseg_data(filepath):
    """
    Load LSEG adaptation market growth data.

    Parameters:
    -----------
    filepath : str
        Path to the LSEG CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with market cap, revenue, and bond issuance data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded LSEG data for {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    return df


def load_artemis_data(filepath):
    """
    Load Artemis catastrophe bonds data and convert cumulative to yearly issuance.

    Parameters:
    -----------
    filepath : str
        Path to the Artemis CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with year, transactions, and yearly issuance (issued_m)
    """
    df = pd.read_csv(filepath)

    # Calculate yearly issuance from cumulative values
    # For the first year, yearly issuance = cumulative value
    # For subsequent years, yearly issuance = current cumulative - previous cumulative
    df['issued_m'] = df['cum_issued_m'].diff().fillna(df['cum_issued_m'].iloc[0])

    print(f"Loaded Artemis cat bonds data for {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    print(f"Converted cumulative to yearly issuance values")

    return df


def load_swiss_re_data(filepath):
    """
    Load Swiss RE catastrophe bonds data.

    Parameters:
    -----------
    filepath : str
        Path to the Swiss RE CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with year, issued_b, and outstanding_b
    """
    df = pd.read_csv(filepath)
    print(f"Loaded Swiss RE cat bonds data for {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    return df


def create_adaptation_solutions_chart(df):
    """
    Create grouped bar chart showing market capitalization and revenue
    for adaptation solutions companies.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with year, market_cap_b, and revenue_b columns

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter to years with both market cap and revenue data
    plot_df = df[df['market_cap_b'].notna() & df['revenue_b'].notna()].copy()

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Bar width and positions
    width = 0.35
    x = range(len(plot_df))
    x_labels = plot_df['year'].astype(int).astype(str)

    # Create grouped bars
    bars1 = ax.bar([i - width/2 for i in x], plot_df['market_cap_b'],
                   width, label='Market Capitalisation',
                   color='#24FD8C', alpha=0.9)
    bars2 = ax.bar([i + width/2 for i in x], plot_df['revenue_b'],
                   width, label='Revenue',
                   color='#3A4E5C', alpha=0.9)

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', labelsize=11)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(0, plot_df[['market_cap_b', 'revenue_b']].max().max() * 1.15)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, frameon=False)

    ax.annotate(">TRILLION $ REVENUE\n5.1% CAGR",
            xy=(6, 1400),
            fontsize=10, ha='left', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#E74C3C', alpha=0.9))

    return fig


def create_green_bonds_chart(df):
    """
    Create stacked bar chart showing adaptation-related green bond issuance
    by economy type (advanced, emerging/developing, supranational).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with year and bond issuance columns

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter to years with bond issuance data
    bond_cols = ['adaption_bond_issuance_b_advanced',
                 'adaption_bond_issuance_b_emerging',
                 'adaption_bond_issuance_b_supranational']
    plot_df = df[df[bond_cols].notna().any(axis=1)].copy()

    # Fill NaN with 0 for stacking
    plot_df[bond_cols] = plot_df[bond_cols].fillna(0)

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Bar width
    width = 0.6
    x = range(len(plot_df))
    x_labels = plot_df['year'].astype(int).astype(str)

    # Create stacked bars
    bars1 = ax.bar(x, plot_df['adaption_bond_issuance_b_advanced'],
                   width, label='Advanced Economies',
                   color='#3A4E5C', alpha=0.9)
    bars2 = ax.bar(x, plot_df['adaption_bond_issuance_b_emerging'],
                   width, bottom=plot_df['adaption_bond_issuance_b_advanced'],
                   label='Emerging and Developing Economies',
                   color='#7B6B8F', alpha=0.9)
    bars3 = ax.bar(x, plot_df['adaption_bond_issuance_b_supranational'],
                   width,
                   bottom=plot_df['adaption_bond_issuance_b_advanced'] + plot_df['adaption_bond_issuance_b_emerging'],
                   label='Supranational',
                   color='#8FA9C7', alpha=0.9)

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', labelsize=11)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    total = (plot_df['adaption_bond_issuance_b_advanced'] +
             plot_df['adaption_bond_issuance_b_emerging'] +
             plot_df['adaption_bond_issuance_b_supranational'])
    ax.set_ylim(0, total.max() * 1.15)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, frameon=False)

    return fig


def create_swiss_re_cat_bonds_chart(df):
    """
    Create stacked bar chart showing Swiss RE catastrophe bond issuance
    and outstanding notional amounts.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with year, issued_b, and outstanding_b columns

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate outstanding from previous years
    df = df.copy()

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Bar width and positions
    width = 0.6
    x = range(len(df))
    x_labels = [str(int(year)) if year != 2025 else '2025\nYTD' for year in df['year']]

    # Create stacked bars
    bars1 = ax.bar(x, df['issued_b'], width,
                   label='Issued', color='#7B3F8F', alpha=0.95)
    bars2 = ax.bar(x, df['outstanding_b'], width,
                   bottom=df['issued_b'],
                   label='Outstanding from previous years',
                   color='#D291BC', alpha=0.85)

    # Add total value labels on top of each bar
    for i, (idx, row) in enumerate(df.iterrows()):
        total = row['outstanding_b'] + row['issued_b']
        ax.text(i, total + 1.5, f'{total:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add individual segment labels
    for i, (idx, row) in enumerate(df.iterrows()):
        # Label for issued amount (bottom segment)
        ax.text(i, row['issued_b'] / 2, f'{row["issued_b"]:.1f}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')
        # Label for outstanding from previous years (top segment)
        if row['outstanding_b'] > 2:  # Only label if segment is large enough
            ax.text(i, row['issued_b'] + row['outstanding_b'] / 2,
                    f'{row["outstanding_b"]:.1f}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white')

    # Calculate CAGR
    years = len(df) - 1
    start_value = df['outstanding_b'].iloc[0]
    end_value = df['outstanding_b'].iloc[-1]
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100

    # Add CAGR annotation
    ax.annotate('13.4% CAGR\n(4.5 years)',
                xy=(0.5, 0.9), xycoords='axes fraction',
                fontsize=11, ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#3A4E5C', linewidth=2))

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylabel('', fontsize=12)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(0, 60)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, frameon=False)

    return fig


def create_cat_bonds_chart(df):
    """
    Create bar chart with line overlay showing catastrophe bond transactions
    and issuance over time.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with year, transactions, and issued_m columns

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up the plot with dual y-axes
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 9))

    # Create second y-axis for issuance
    ax2 = ax.twinx()

    # Bar chart for transactions on primary axis
    width = 0.7
    bars = ax.bar(df['year'], df['transactions'],
                  width, color='#3A4E5C', alpha=0.85,
                  label='Transactions', zorder=2)

    # Line chart for issuance on secondary axis
    line = ax2.plot(df['year'], df['issued_m'] / 1000,  # Convert to billions
                    color='#2EF5A6', linewidth=3, marker='o',
                    markersize=6, label='Issuance', zorder=3)

    # Format primary y-axis (left - transactions)
    ax.set_ylabel('Number of Deals', fontsize=12)
    ax.tick_params(axis='y', labelsize=11, colors='#3A4E5C')
    ax.spines['left'].set_color('#3A4E5C')
    ax.set_ylim(0, df['transactions'].max() * 1.15)

    # Format secondary y-axis (right - issuance in billions)
    ax2.set_ylabel('Issuance (US$ Billions)', fontsize=12, color='#3A4E5C')
    ax2.tick_params(axis='y', labelsize=11, colors='#3A4E5C')
    ax2.spines['right'].set_color('#24FD8C')
    ax2.set_ylim(0, (df['issued_m'].max() / 1000) * 1.15)

    # Format x-axis
    ax.set_xlim(df['year'].min() - 0.5, df['year'].max() + 0.5)
    ax.tick_params(axis='x', labelsize=11)

    # Modify tick labels to add YTD to 2025
    labels = [label.get_text() if label.get_text() != '2025' else '2025\nYTD'
              for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    # Add grid
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Add legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
             loc='upper left', fontsize=11, frameon=True,
             facecolor='white', edgecolor='#DDDDDD')

    return fig


def load_hvac_stock_data(filepath):
    """
    Load HVAC stock price data and S&P 500 data.

    Parameters:
    -----------
    filepath : str
        Path to the HVAC stock prices CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with month, stock prices, and S&P 500 data
    """
    df = pd.read_csv(filepath)

    # Remove commas from S&P 500 values and convert to float
    df['s_p_500'] = df['s_p_500'].str.replace(',', '').astype(float)

    # Convert month to datetime
    df['month'] = pd.to_datetime(df['month'], format='%d-%b-%y')

    # Sort by date (oldest first)
    df = df.sort_values('month').reset_index(drop=True)

    print(f"Loaded HVAC stock data from {df['month'].min().strftime('%b %Y')} to {df['month'].max().strftime('%b %Y')}")

    return df


def create_hvac_indexed_chart(df):
    """
    Create indexed chart showing HVAC stock performance vs S&P 500.
    All values normalized to 100 at the start date (January 2010).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with month, trane, carrier, lennox, johnson, s_p_500 columns

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 9))

    # Normalize all values to start at 100
    stocks = ['trane', 'johnson', 'lennox', 's_p_500']
    indexed_df = df.copy()

    for stock in stocks:
        # Find first non-null value for this stock
        first_value = indexed_df[stock].dropna().iloc[0]
        indexed_df[f'{stock}_indexed'] = (indexed_df[stock] / first_value) * 100

    # Calculate mean of HVAC companies
    indexed_df['hvac_mean_indexed'] = indexed_df[['trane_indexed', 'johnson_indexed', 'lennox_indexed']].mean(axis=1)

    # Define colors
    colors = {
        'hvac_mean': '#24FD8C',  # Green
        's_p_500': '#7F8C8D'  # Gray
    }

    labels = {
        'hvac_mean': 'HVAC Companies (Trane, Johnson Controls, Lennox)',
        's_p_500': 'S&P 500'
    }

    # Plot S&P 500 first (so it's in the background)
    ax.plot(indexed_df['month'], indexed_df['s_p_500_indexed'],
            color=colors['s_p_500'], linewidth=2.5, alpha=0.7,
            label=labels['s_p_500'], zorder=1)

    # Plot HVAC mean
    ax.plot(indexed_df['month'], indexed_df['hvac_mean_indexed'],
            color=colors['hvac_mean'], linewidth=3,
            label=labels['hvac_mean'], zorder=2)

    # Format axes
    ax.tick_params(axis='both', labelsize=11, length=0)
    ax.set_ylim(0, 1500)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add horizontal grid lines
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color="#8E8E8E", zorder=0)
    ax.set_axisbelow(True)

    # Calculate growth rates
    hvac_growth = indexed_df['hvac_mean_indexed'].iloc[-1] - 100
    sp500_growth = indexed_df['s_p_500_indexed'].iloc[-1] - 100

    # Add legend
    # ax.legend(loc='upper left', fontsize=11, frameon=True,
    #          facecolor='white', edgecolor='#DDDDDD')

    # Annotate S&P 500 using axes fraction coordinates (0-1 range)
    # xy=(0.5, 0.3) means 50% across, 30% up from bottom
    ax.annotate(f"S&P 500\n+{sp500_growth:.0f}% GROWTH",
                xy=(0.85, 0.2), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                color='#7F8C8D')

    # Annotate HVAC Companies
    ax.annotate(f"HVAC COMPANIES\n+{hvac_growth:.0f} GROWTH%",
                xy=(0.79, 0.81), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                color='#24FD8C')

    return fig


def main():
    """
    Main execution function.
    """
    # File paths
    lseg_file = 'data/adaptation_market_growth/lseg_adaptation_growth.csv'
    artemis_file = 'data/adaptation_market_growth/artemis_cat_bonds.csv'
    swiss_re_file = 'data/adaptation_market_growth/swiss_re_cat_bonds.csv'
    hvac_file = 'data/adaptation_market_growth/hvac_stock_proces.csv'
    figures_dir = 'figures'

    # Create output directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    print('\n=== Loading Adaptation Market Data ===\n')

    # Load data
    lseg_df = load_lseg_data(lseg_file)
    artemis_df = load_artemis_data(artemis_file)
    swiss_re_df = load_swiss_re_data(swiss_re_file)

    # Create adaptation solutions chart
    print('\n=== Creating Adaptation Solutions Chart ===\n')
    fig1 = create_adaptation_solutions_chart(lseg_df)

    format_plot_title(plt.gca(),
                     '',
                     'CLIMATE ADAPTATION MARKET (US$ BILLION)',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note='DATA: LONDON STOCK EXCHANGE GROUP (2025)',
                         analysis_date=datetime.now())

    save_path1 = os.path.join(figures_dir, 'adaptation_solutions_size.png')
    save_plot(fig1, save_path1)
    print(f'Adaptation solutions chart saved to {save_path1}')

    # Create green bonds chart
    print('\n=== Creating Green Bonds Chart ===\n')
    fig2 = create_green_bonds_chart(lseg_df)

    format_plot_title(plt.gca(),
                     '',
                     'ADAPTATION-RELATED GREEN BOND ISSUANCE (US$ BILLION)',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note='DATA: LONDON STOCK EXCHANGE GROUP (2025)',
                         analysis_date=datetime.now())

    save_path2 = os.path.join(figures_dir, 'adaptation_green_bonds.png')
    save_plot(fig2, save_path2)
    print(f'Green bonds chart saved to {save_path2}')

    # Create catastrophe bonds chart
    print('\n=== Creating Catastrophe Bonds Chart ===\n')
    fig3 = create_cat_bonds_chart(artemis_df)

    format_plot_title(plt.gca(),
                     '',
                     'CATASTROPHE BONDS ISSUANCE AND NUMBER OF DEALS',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note='DATA: ARTEMIS DEAL DIRECTORY',
                         analysis_date=datetime.now())

    save_path3 = os.path.join(figures_dir, 'catastrophe_bonds_issuance.png')
    save_plot(fig3, save_path3)
    print(f'Catastrophe bonds chart saved to {save_path3}')

    # Create Swiss RE catastrophe bonds chart
    print('\n=== Creating Swiss RE Catastrophe Bonds Chart ===\n')
    fig4 = create_swiss_re_cat_bonds_chart(swiss_re_df)

    format_plot_title(plt.gca(),
                     '',
                     'CATASTROPHE BONDS: ISSUED VS OUTSTANDING NOTIONAL (US$ BILLION)',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note='DATA: SWISS RE CAPITAL MARKETS DEAL DATABASE AS OF 30 JUNE 2025')

    save_path4 = os.path.join(figures_dir, 'swiss_re_cat_bonds.png')
    save_plot(fig4, save_path4)
    print(f'Swiss RE catastrophe bonds chart saved to {save_path4}')

    # Create HVAC stock performance chart
    print('\n=== Creating HVAC Stock Performance Chart ===\n')
    hvac_df = load_hvac_stock_data(hvac_file)
    fig5 = create_hvac_indexed_chart(hvac_df)

    format_plot_title(plt.gca(),
                     '',
                     'HVAC COMPANIES STOCK PERFORMANCE VS S&P 500 (INDEXED)',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note='DATA: YAHOO FINANCE. ALL VALUES INDEXED TO 100 AT JANUARY 2010.',
                         analysis_date=datetime.now())

    save_path5 = os.path.join(figures_dir, 'hvac_stock_performance.png')
    save_plot(fig5, save_path5)
    print(f'HVAC stock performance chart saved to {save_path5}')

    # Print summary statistics
    print('\n=== Summary Statistics ===\n')
    print('Adaptation Solutions (latest year with data):')
    latest_solutions = lseg_df[lseg_df['market_cap_b'].notna()].iloc[-1]
    print(f'  Year: {int(latest_solutions["year"])}')
    print(f'  Market Cap: ${latest_solutions["market_cap_b"]:.1f}B')
    if pd.notna(latest_solutions['revenue_b']):
        print(f'  Revenue: ${latest_solutions["revenue_b"]:.1f}B')

    print('\nGreen Bonds (latest year):')
    bond_cols = ['adaption_bond_issuance_b_advanced',
                 'adaption_bond_issuance_b_emerging',
                 'adaption_bond_issuance_b_supranational']
    latest_bonds = lseg_df[lseg_df[bond_cols].notna().any(axis=1)].iloc[-1]
    print(f'  Year: {int(latest_bonds["year"])}')
    total_issuance = latest_bonds[bond_cols].sum()
    print(f'  Total Issuance: ${total_issuance:.1f}B')

    print('\nCatastrophe Bonds (latest year):')
    latest_cat = artemis_df.iloc[-1]
    print(f'  Year: {int(latest_cat["year"])}')
    print(f'  Issuance: ${latest_cat["issued_m"] / 1000:.1f}B')
    print(f'  Number of Deals: {int(latest_cat["transactions"])}')

    print('\nHVAC Stock Performance (Jan 2010 - Latest):')

    # Calculate individual stock performance
    stocks = ['trane', 'johnson', 'lennox', 's_p_500']
    growth_rates = []

    for stock in stocks:
        if hvac_df[stock].notna().any():
            first_value = hvac_df[stock].dropna().iloc[0]
            last_value = hvac_df[stock].dropna().iloc[-1]
            growth_pct = ((last_value / first_value) - 1) * 100

            stock_name = {
                'trane': 'Trane Technologies',
                'johnson': 'Johnson Controls',
                'lennox': 'Lennox International',
                's_p_500': 'S&P 500'
            }[stock]

            print(f'  {stock_name}: +{growth_pct:.0f}% (${first_value:.2f} → ${last_value:.2f})')

            if stock != 's_p_500':
                growth_rates.append(growth_pct)

    # Calculate and print average HVAC performance
    if growth_rates:
        avg_growth = sum(growth_rates) / len(growth_rates)
        print(f'\n  Average HVAC Performance: +{avg_growth:.0f}%')

    print('\n=== Script Complete ===\n')


if __name__ == '__main__':
    main()
