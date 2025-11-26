# Deep Sky Data Science

This repository contains data science tools, research pipelines, and interactive dashboards for climate data analysis and visualization. The platform processes climate and environmental data from various sources to support Deep Sky's research and client services.

## Architecture

The platform consists of several interconnected components:

**Data Collection**: Raw climate data is collected from multiple sources including ERA5, NOAA, AVISO, and CDR.fyi APIs.

**Data Processing**: Raw data is cleaned, validated, and transformed into analysis-ready formats using Python and scientific computing libraries.

**Research Reports**: Climate analysis reports are generated using UNSEEN (UNprecedented Simulated Extremes using ENsembles) methodology for extreme weather events.

**Interactive Dashboards**: Two main dashboard systems serve processed data through AWS Lambda functions and S3 storage:
- Deep Sky Research Dashboard: Climate indicators and trends
- CDR Market Dashboard: Carbon dioxide removal market analytics

**Data Visualization**: Interactive charts and maps are generated for web integration using JavaScript and Python plotting libraries.

## Project Structure

```
Deep_Sky_Data_Science/
├── cdr_dashboard/           # CDR market data dashboard
│   ├── config/             # Dashboard configuration
│   ├── data/               # Processed CDR market data
│   ├── deployment/         # AWS deployment scripts
│   ├── lambdas/           # Lambda function handlers
│   ├── retrieval/         # Data collection scripts
│   └── utils/             # Utility functions
├── dsr_dashboard/          # Deep Sky Research dashboard
│   ├── config/            # Dashboard configuration
│   ├── data/              # Processed climate data
│   ├── deployment/        # AWS deployment scripts
│   ├── lambdas/          # Lambda function handlers
│   ├── retrieval/        # Climate data retrievers
│   └── utils/            # Utility functions
├── reports/               # UNSEEN climate analysis reports
│   ├── heatwave/         # Heat wave report
│   ├── hurricane/        # Hurricane report
│   ├── wildfire/         # First wildfire report
│   └── wildfires_2025/   # Second wildfire report
├── scrapers/             # Data collection scripts
├── viz/                  # Data visualization tools
└── exploratory_projects/ # Research and analysis projects
```

## Key Components

**Dashboard Systems**: 
- `cdr_dashboard/`: Carbon dioxide removal market analytics and visualization
- `dsr_dashboard/`: Climate research data dashboard with multiple climate indicators

**Research Pipeline**: 
- `reports/`: Scripts for exploring, analysing, and modeling data on climate events and impacts for publishing on research page
- `scrapers/`: Automated data collection from various climate data sources
- `viz/`: Interactive visualization and plotting 

## Prerequisites

**Python Environment**:
- Python 3.8 or higher
- Scientific computing libraries (numpy, pandas, xarray, netCDF4)
- AWS SDK (boto3) for cloud deployment

**AWS Infrastructure**:
- AWS CLI configured with appropriate S3 permissions
- Lambda deployment permissions
- ECR repository access for container deployments

**Development Tools**:
- Docker for containerized deployments
- Terraform for infrastructure management
- Git for version control

## Local Development

### Setup
```bash
# Install core dependencies
pip install -r dsr_dashboard/requirements.txt

# Configure AWS credentials
aws configure sso --profile deep-sky-test
aws configure sso --profile deep-sky-prod
```

### Testing
```bash
# Run dashboard tests
cd dsr_dashboard
python -m pytest tests/

# Run individual data retrievers
python retrieval/co2_ppm.py
python retrieval/era5_processor.py
```

### Local Dashboard Server
```bash
# Start local development server
python dsr_dashboard/deployment/local_server.py
```

## Deployment

### Dashboard Deployment
```bash
# Deploy to test environment
python dsr_dashboard/deployment/deploy.py test

# Deploy to production
python dsr_dashboard/deployment/deploy.py prod
```

### CDR Dashboard Deployment
```bash
# Deploy CDR dashboard
python cdr_dashboard/deployment/deploy.py test
python cdr_dashboard/deployment/deploy.py prod
```

## Data Sources

**Climate Data**:
- ERA5 reanalysis data (temperature, precipitation, fire weather index)
- NOAA climate indicators and billion-dollar disasters
- AVISO sea level rise measurements
- Berkeley Earth temperature records

**Market Data**:
- CDR.fyi carbon removal marketplace data
- Carbon pricing information
- Insurance and financial climate impact data

**Geospatial Data**:
- Shapefiles for North America, US states, and counties
- Power grid outage data
- Wildfire and extreme weather event databases
