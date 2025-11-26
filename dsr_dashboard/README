# Deep Sky Research Dashboard

This directory contains code for retrieving and delivering data displayed in the Deep Sky Research Dashboard. The system uses AWS Lambda functions which run daily to pull data from various sources and store it as CSV files in an S3 bucket, which are then accessed by the Deep Sky Research website through presigned URLs.

## System Architecture

### Data Flow
1. Lambda functions run on a weekly schedule to retrieve data
2. Data is processed and stored as CSV files in S3
3. Website requests data via presigned URL API
4. Lambda generates temporary URLs for secure data access

### Key Components
- Data Retrieval Scripts (`retrieval/`)
- Lambda Function Handlers (`lambdas/`)
- Deployment Scripts (`deployment/`)
- Infrastructure Code (`../../data-science-infrastructure/terraform/`)
- Unit Tests (`tests/`)

## Datasets

The dashboard currently manages these datasets:
- ERA5 Climate Data (air/sea surface temperature)
- Aviso Sea Level Rise
- NOAA Billion Dollar Disasters
- CO2 Atmospheric Concentrations
- Home Insurance Premiums
- Ocean pH (disabled at the moment)

Dataset configurations are managed in `config/dataset_dir.json`.

## Local Development

Run all the following commands from the `Deep_Sky_Data_Science/dashboard/` directory.

### Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:

```bash
aws configure sso --profile test
aws configure sso --profile prod
```

### Testing
Run the test suite to verify data retrieval and processing:

```bash
python -m pytest tests/
```

Key test files:
- `test_data_retrievers.py`: Tests individual data source retrievers
- `test_end_to_end.py`: Tests complete data pipeline
- `test_era5_utils.py`: Tests ERA5 data processing utilities

### Local Data Updates
Test data retrieval locally using:

```bash
python lambdas/data_updater.py
```

or individual retrieval scripts in the `retrieval/` directory. For example:

```bash
python retrieval/noaa_billion_dollar_disasters.py
```


## Deployment Strategy

### Prerequisites
- AWS CLI configured with appropriate credentials
- Docker installed locally
- Terraform installed
- Access to AWS test and prod environments

### Environment Setup

- **Test Environment**
  - Used for testing changes
  - Connects to test Webflow site
  - AWS Profile: `deep-sky-test`
  - S3 Bucket: `dashboard-metrics-test-{account-id}`

- **Production Environment**
  - Serves the public website
  - AWS Profile: `deep-sky-prod`
  - S3 Bucket: `dashboard-metrics-prod-{account-id}`

### Deployment Process

1. **Local Development**
   ```bash
   # Make and test changes locally
   python -m pytest tests/
   ```

2. **Deploy to Test Environment**
   ```bash
   # Deploy to test environment
   python deployment/deploy.py test
   ```

   # Verify changes in test environment
   # Test dashboard functionality on test Webflow site
   # https://www.deepskyclimate.com/utility-pages/mdk-dashboard-charts-testing
   ```

3. **Check Test Deployment**

   - Test dasboard functionality on test Webflow site
      - https://www.deepskyclimate.com/utility-pages/mdk-dashboard-charts-testing
   - Test data retrieval Lambda:

   ```bash
   aws lambda invoke \
   --function-name dashboard_update_test \
   --payload '{}' \
   response.json

   cat response.json
   ```

   - Check retreival logs in S3 bucket /logs/retrieval.log

3. **Production Deployment**
   ```bash
   # After successful testing, deploy to production
   python deployment/deploy.py prod
   ```


## Troubleshooting

### Common Issues
1. Lambda Deployment Failures
   - Check ECR repository permissions
   - Verify Lambda IAM roles
   - Review CloudWatch logs

2. Data Retrieval Issues
   - Verify API credentials in AWS Secrets Manager
   - Check source data availability
   - Review error logs in CloudWatch

### Deployment Script

The `deploy.py` script handles:
- Using the correct AWS profile for each environment
- Building and pushing the Lambda container
- Updating Lambda functions
- Uploading chart templates and configuration files to S3

### Best Practices

1. **Always Test First**: Deploy changes to test environment before production

2. **Version Control**:
   - Commit all code changes to git
   - Use descriptive commit messages
   - Create Pull Requests for significant changes

3. **Monitoring**:
   - Check Lambda logs after deployments
   - Monitor CloudWatch metrics
   - Verify dashboard functionality after changes
   - Check S3 bucket logs/retrieval.log for retrieval logs

4. **Security**:
   - Keep AWS credentials secure
   - Use separate AWS profiles for each environment
   - Never commit sensitive data to git

5. **Documentation**:
   - Document significant changes
   - Update README when adding new features
   - Include deployment notes in commit messages

When updating the system:

1. Make code changes and test locally
2. Run test suite: `python -m pytest tests/`
3. Update infrastructure if needed (Terraform)
4. Deploy code changes:
   - Build and push container
   - Update Lambda functions
5. Test deployed changes
6. Monitor logs and metrics

Remember to:
- Keep test and prod environments in sync
- Document significant changes
- Monitor AWS costs
- Review security configurations regularly

## Dashboard Refresh Schedule

The dashboard data is automatically updated on a weekly schedule:
- Trigger: Every Sunday at 7:00 PM EST (Monday 00:00 UTC)
- Lambda function: `dashboard_update_{environment}`
- Schedule defined in: `data-science-infrastructure/terraform/test/lambda.tf`

The schedule can be modified by updating the `schedule_expression` in the `aws_cloudwatch_event_rule` resource. The current cron expression `cron(0 0 ? * MON *)` breaks down as:
- `0` - At minute 0
- `0` - At hour 0 (midnight UTC)
- `?` - Any day of month
- `*` - Any month
- `MON` - On Monday
- `*` - Any year

Note: AWS EventBridge uses UTC time. EST is UTC-5 (or UTC-4 during daylight savings).