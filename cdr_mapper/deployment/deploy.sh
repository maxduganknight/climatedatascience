#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get environment from command line argument
ENVIRONMENT="${1:-test}"  # Default to test if not specified

# Validate environment
if [[ "$ENVIRONMENT" != "test" && "$ENVIRONMENT" != "prod" ]]; then
    echo "Error: Environment must be 'test' or 'prod'"
    echo "Usage: ./deploy.sh [test|prod]"
    exit 1
fi

# Use environment-specific AWS profile
AWS_PROFILE="${ENVIRONMENT}"
export AWS_PROFILE

echo "=================================================="
echo "Deploying CDR Mapper to ${ENVIRONMENT} environment"
echo "=================================================="

# Get AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)

echo "AWS Account: ${ACCOUNT_ID}"
echo "AWS Region: ${AWS_REGION}"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the cdr_mapper root directory (parent of deployment dir)
CDR_MAPPER_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check that we're in a git repository
if ! git -C "$CDR_MAPPER_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git -C "$CDR_MAPPER_DIR" diff-index --quiet HEAD --; then
    echo "Warning: You have uncommitted changes"
    echo "Current changes:"
    git -C "$CDR_MAPPER_DIR" status --short
    echo ""
    read -p "Continue with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Get current git branch and commit
GIT_BRANCH=$(git -C "$CDR_MAPPER_DIR" rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git -C "$CDR_MAPPER_DIR" rev-parse --short HEAD)

echo "Deploying from branch: ${GIT_BRANCH}"
echo "Current commit: ${GIT_COMMIT}"
echo ""

# Get EC2 instance ID from Terraform
echo "Getting EC2 instance ID from Terraform..."
cd ../../datascience-infra/terraform/${ENVIRONMENT}

# Check if Terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo "Error: Terraform state not found. Have you run 'terraform apply' yet?"
    exit 1
fi

INSTANCE_ID=$(terraform output -raw cdr_mapper_instance_id 2>/dev/null)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
    echo "Error: CDR Mapper instance not found in Terraform state"
    echo "Please deploy infrastructure first with 'terraform apply'"
    exit 1
fi

echo "Instance ID: ${INSTANCE_ID}"
echo ""

# Get S3 bucket name while still in terraform directory
S3_BUCKET=$(terraform output -raw cdr_mapper_s3_bucket 2>/dev/null)

if [ -z "$S3_BUCKET" ] || [ "$S3_BUCKET" = "null" ]; then
    echo "Error: CDR Mapper S3 bucket not found in Terraform state"
    exit 1
fi

echo "S3 Bucket: ${S3_BUCKET}"
echo ""

# Check instance state
INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text)

if [ "$INSTANCE_STATE" != "running" ]; then
    echo "Error: Instance is not running (current state: ${INSTANCE_STATE})"
    echo "Please start the instance first"
    exit 1
fi

echo "Instance state: ${INSTANCE_STATE}"
echo ""

# Go back to cdr_mapper directory for creating tarball
cd "$CDR_MAPPER_DIR"

# Create tarball of application files
echo "Creating application tarball..."
TARBALL="/tmp/cdr-mapper-${GIT_COMMIT}.tar.gz"

# Include all necessary files, including Git LFS cached data
tar -czf "$TARBALL" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    .

echo "Tarball created: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo ""

# Upload tarball to EC2 via S3 (temporary storage)
echo "Uploading tarball to S3..."
S3_KEY="deployments/cdr-mapper-${GIT_COMMIT}.tar.gz"
aws s3 cp "$TARBALL" "s3://${S3_BUCKET}/${S3_KEY}"
echo ""

# Create deployment script to run on EC2
DEPLOY_SCRIPT=$(cat <<EOFSCRIPT
#!/bin/bash
set -e

echo "=== Starting CDR Mapper Deployment ==="
date

# Download and extract application files
echo "Downloading application from S3..."
aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" /tmp/cdr-mapper.tar.gz

echo "Extracting application files..."
rm -rf /opt/cdr-mapper
mkdir -p /opt/cdr-mapper
tar -xzf /tmp/cdr-mapper.tar.gz -C /opt/cdr-mapper
rm /tmp/cdr-mapper.tar.gz

cd /opt/cdr-mapper

# Build new Docker image
echo "Building Docker image..."
docker build -t cdr-mapper -f deployment/Dockerfile .

if [ \$? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

# Stop old container
echo "Stopping old container..."
docker stop cdr-mapper || true
docker rm cdr-mapper || true

# Start new container
echo "Starting new container..."
docker run -d \
  --name cdr-mapper \
  --restart unless-stopped \
  -p 8501:8501 \
  -e CDR_MAPPER_LOG_BUCKET=${S3_BUCKET} \
  cdr-mapper

# Wait for container to be healthy
echo "Waiting for container to start..."
sleep 5

# Check if container is running
if docker ps | grep -q cdr-mapper; then
    echo "✓ Container is running"
    docker ps | grep cdr-mapper
else
    echo "✗ Container failed to start"
    echo "Container logs:"
    docker logs cdr-mapper
    exit 1
fi

echo "=== Deployment Complete ==="
date
EOFSCRIPT
)

# Clean up local tarball
rm -f "$TARBALL"

# Execute deployment via SSM
echo "Executing deployment on EC2 instance..."
echo ""

# Create a temporary JSON file for the parameters
PARAMS_FILE=$(mktemp)
cat > "$PARAMS_FILE" <<EOF
{
  "commands": [
    $(echo "$DEPLOY_SCRIPT" | jq -Rs .)
  ]
}
EOF

COMMAND_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters file://"$PARAMS_FILE" \
    --output text \
    --query 'Command.CommandId')

rm -f "$PARAMS_FILE"

echo "Command ID: ${COMMAND_ID}"
echo ""
echo "Waiting for deployment to complete..."

# Wait for command to complete
while true; do
    STATUS=$(aws ssm get-command-invocation \
        --command-id "$COMMAND_ID" \
        --instance-id "$INSTANCE_ID" \
        --query 'Status' \
        --output text)

    if [ "$STATUS" = "Success" ]; then
        echo ""
        echo "✓ Deployment succeeded!"
        break
    elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ] || [ "$STATUS" = "TimedOut" ]; then
        echo ""
        echo "✗ Deployment failed with status: $STATUS"
        echo ""
        echo "Command output:"
        aws ssm get-command-invocation \
            --command-id "$COMMAND_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardOutputContent' \
            --output text
        echo ""
        echo "Error output:"
        aws ssm get-command-invocation \
            --command-id "$COMMAND_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardErrorContent' \
            --output text
        exit 1
    else
        echo -n "."
        sleep 2
    fi
done

# Get deployment output
echo ""
echo "Deployment output:"
echo "-------------------"
aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text

# Get application URL
APP_URL=$(terraform output -raw cdr_mapper_url 2>/dev/null)

echo ""
echo "=================================================="
echo "Deployment complete!"
echo "Application URL: ${APP_URL}"
echo "Environment: ${ENVIRONMENT}"
echo "Deployed commit: ${GIT_COMMIT}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Verify the application is working: ${APP_URL}"
echo "2. Check container logs: ssh ec2-user@<instance-ip> 'docker logs -f cdr-mapper'"
echo "3. Monitor CloudWatch metrics"
echo ""
