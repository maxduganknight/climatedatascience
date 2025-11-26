#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get environment from command line argument
ENVIRONMENT="${1:-test}"  # Default to test if not specified

# Use environment-specific AWS profile
AWS_PROFILE="${ENVIRONMENT}"
export AWS_PROFILE

# Get AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
LAMBDA_REPO="dashboard-lambda-${ENVIRONMENT}"
BUCKET_NAME="dashboard-metrics-${ENVIRONMENT}-${ACCOUNT_ID}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the dashboard root directory (parent of deployment dir)
DASHBOARD_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Function to wait for Lambda to be ready
wait_for_lambda() {
    local function_name=$1
    echo "Waiting for $function_name to be ready..."
    while true; do
        status=$(aws lambda get-function --function-name $function_name --query 'Configuration.LastUpdateStatus' --output text)
        if [ "$status" = "Successful" ]; then
            echo "$function_name is ready"
            break
        elif [ "$status" = "Failed" ]; then
            echo "$function_name update failed"
            exit 1
        fi
        echo "Current status: $status. Waiting..."
        sleep 5
    done
}

# Build container
echo "Building Lambda container..."
docker build --platform linux/amd64 --provenance=false \
    -t ${LAMBDA_REPO} \
    -f deployment/Dockerfile .

# Only proceed if build was successful
if [ $? -eq 0 ]; then
    # Authenticate Docker to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
      docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    # Tag and push container
    docker tag ${LAMBDA_REPO}:latest ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest
    docker push ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest

    # Update first Lambda function
    aws lambda update-function-code \
        --function-name dashboard_update_${ENVIRONMENT} \
        --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest

    wait_for_lambda "dashboard_update_${ENVIRONMENT}"

    # Update second Lambda function
    aws lambda update-function-code \
        --function-name ds-presigned-url-generator-${ENVIRONMENT} \
        --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest

    wait_for_lambda "ds-presigned-url-generator-${ENVIRONMENT}"

    echo "Lambda functions updated with new container image"

    aws lambda update-function-code \
        --function-name dashboard-dataset-versions-${ENVIRONMENT} \
        --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest

    wait_for_lambda "dashboard-dataset-versions-${ENVIRONMENT}"

    echo "Dataset versions Lambda function updated"


    # Update RSS feed Lambda function
    aws lambda update-function-code \
        --function-name dashboard-rss-feed-${ENVIRONMENT} \
        --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest

    wait_for_lambda "dashboard-rss-feed-${ENVIRONMENT}"

    echo "RSS feed Lambda function updated"

    # Upload chart templates and dataset config to S3
    echo "Uploading chart templates and dataset config to S3..."
    TEMPLATES_PATH="${DASHBOARD_DIR}/webflow/chart_templates.js"
    CONFIG_PATH="${DASHBOARD_DIR}/config/dataset_dir.json"

    if [ -f "$TEMPLATES_PATH" ] && [ -f "$CONFIG_PATH" ]; then
        # Upload chart templates
        aws s3 cp "$TEMPLATES_PATH" "s3://${BUCKET_NAME}/viz/chart_templates.js" \
            --content-type "application/javascript" \
            --profile ${ENVIRONMENT}
        
        # Upload dataset config
        aws s3 cp "$CONFIG_PATH" "s3://${BUCKET_NAME}/processed/dataset_dir.json" \
            --content-type "application/json" \
            --profile ${ENVIRONMENT}
        
        echo "Files uploaded successfully"
    else
        echo "Error: Required files not found"
        [ ! -f "$TEMPLATES_PATH" ] && echo "Missing: ${TEMPLATES_PATH}"
        [ ! -f "$CONFIG_PATH" ] && echo "Missing: ${CONFIG_PATH}"
        exit 1
    fi

    echo "Deployment complete!"
else
    echo "Docker build failed. Exiting without pushing or updating Lambda functions."
    exit 1
fi