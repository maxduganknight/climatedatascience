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
LAMBDA_REPO="cdr-lambda-${ENVIRONMENT}"
BUCKET_NAME="cdr-dashboard-${ENVIRONMENT}-${ACCOUNT_ID}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the dashboard root directory (parent of deployment dir)
CDR_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Function to wait for Lambda to be ready
wait_for_lambda() {
    local function_name=$1
    echo "Waiting for $function_name to be ready..."
    while true; do
        status=$(aws lambda get-function --function-name $function_name --query 'Configuration.LastUpdateStatus' --output text 2>/dev/null || echo "NotFound")
        if [ "$status" = "Successful" ]; then
            echo "$function_name is ready"
            break
        elif [ "$status" = "Failed" ]; then
            echo "$function_name update failed"
            exit 1
        elif [ "$status" = "NotFound" ]; then
            echo "$function_name not found. Will be created by Terraform."
            break
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

    # Update cdr-dashboard-updater Lambda function
    LAMBDA_NAME="cdr-dashboard-updater-${ENVIRONMENT}"
    echo "Updating Lambda function ${LAMBDA_NAME}..."
    if aws lambda get-function --function-name ${LAMBDA_NAME} >/dev/null 2>&1; then
        # Function exists, update it
        aws lambda update-function-code \
            --function-name ${LAMBDA_NAME} \
            --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest
        wait_for_lambda ${LAMBDA_NAME}
    else
        echo "Function ${LAMBDA_NAME} doesn't exist yet. It should be created by Terraform first."
        echo "Run terraform apply in your infrastructure directory before deploying code."
    fi

    # Repeat the same pattern for the second Lambda function
    LAMBDA_NAME="cdr-ds-presigned-url-generator-${ENVIRONMENT}"
    echo "Updating Lambda function ${LAMBDA_NAME}..."
    if aws lambda get-function --function-name ${LAMBDA_NAME} >/dev/null 2>&1; then
        # Function exists, update it
        aws lambda update-function-code \
            --function-name ${LAMBDA_NAME} \
            --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest
        wait_for_lambda ${LAMBDA_NAME}
    else
        echo "Function ${LAMBDA_NAME} doesn't exist yet. It should be created by Terraform first."
        echo "Run terraform apply in your infrastructure directory before deploying code."
    fi

    # Update missing leads checker Lambda function
    LAMBDA_NAME="missing-leads-checker-${ENVIRONMENT}"
    echo "Updating Lambda function ${LAMBDA_NAME}..."
    if aws lambda get-function --function-name ${LAMBDA_NAME} >/dev/null 2>&1; then
        # Function exists, update it
        aws lambda update-function-code \
            --function-name ${LAMBDA_NAME} \
            --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:latest
        wait_for_lambda ${LAMBDA_NAME}
    else
        echo "Function ${LAMBDA_NAME} doesn't exist yet. It should be created by Terraform first."
        echo "Run terraform apply in your infrastructure directory before deploying code."
    fi

    echo "Lambda functions updated with new container image"
    
    # Create logs directory in S3 bucket if it doesn't exist
    echo "Creating S3 directories if needed..."
    aws s3api head-object --bucket ${BUCKET_NAME} --key logs/ 2>/dev/null || \
        aws s3api put-object --bucket ${BUCKET_NAME} --key logs/ --content-type "application/x-directory"
    
    # Create retrieval.log if it doesn't exist
    echo "Checking for retrieval.log..."
    if ! aws s3api head-object --bucket ${BUCKET_NAME} --key logs/retrieval.log 2>/dev/null; then
        echo "Creating initial retrieval.log file..."
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        echo "CDR Dashboard retrieval log initialized on ${TIMESTAMP}" > /tmp/retrieval.log
        aws s3 cp /tmp/retrieval.log "s3://${BUCKET_NAME}/logs/retrieval.log"
    fi
    
    # Upload chart templates to S3
    echo "Uploading chart templates to S3..."
    TEMPLATES_PATH="${CDR_DIR}/viz/chart_templates.js"
    
    if [ -f "$TEMPLATES_PATH" ]; then
        # Create viz directory in S3 bucket if it doesn't exist
        aws s3api head-object --bucket ${BUCKET_NAME} --key viz/ 2>/dev/null || \
            aws s3api put-object --bucket ${BUCKET_NAME} --key viz/ --content-type "application/x-directory"
        
        # Upload chart templates
        aws s3 cp "$TEMPLATES_PATH" "s3://${BUCKET_NAME}/viz/chart_templates.js" \
            --content-type "application/javascript" \
            --profile ${ENVIRONMENT}
        
        echo "Chart templates uploaded successfully"
    else
        echo "Warning: Chart templates file not found at ${TEMPLATES_PATH}"
        echo "Skipping chart templates upload"
    fi
    
    echo "Deployment complete!"
else
    echo "Docker build failed. Exiting without pushing or updating Lambda functions."
    exit 1
fi