#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== CDR Mapper Instance Setup Started at $(date) ==="

# Get environment from instance tags (passed via Terraform)
ENVIRONMENT="${ENVIRONMENT}"
DNS_NAME="${DNS_NAME}"

echo "Environment: $ENVIRONMENT"
echo "DNS Name: $DNS_NAME"

# Update system
yum update -y

# Install packages
yum install -y docker git git-lfs nginx certbot python3-certbot-nginx

# Enable and start Docker
systemctl enable docker
systemctl start docker
usermod -a -G docker ec2-user

# Configure Git LFS
git lfs install --system

# Clone repository
cd /opt
git clone https://github.com/deepskyclimate/datascience-platform.git
cd datascience-platform/cdr_mapper

# Pull LFS files (cache data)
git lfs pull

# Build Docker image using the Dockerfile from the repo
docker build -t cdr-mapper -f deployment/Dockerfile .

# Run Docker container
docker run -d \
  --name cdr-mapper \
  --restart unless-stopped \
  -p 8501:8501 \
  cdr-mapper

echo "Docker container started"

# Configure Nginx
cat > /etc/nginx/conf.d/cdr-mapper.conf <<NGINX_CONF
server {
    listen 80;
    server_name ${DNS_NAME};

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
NGINX_CONF

# Start Nginx
systemctl enable nginx
systemctl start nginx

echo "Nginx configured"

# Get SSL certificate (wait for DNS to propagate)
echo "Waiting 60 seconds for DNS propagation..."
sleep 60

certbot --nginx -d ${DNS_NAME} \
  --non-interactive \
  --agree-tos \
  -m devops@deepskyclimate.com \
  --redirect || echo "Certbot failed - DNS may not be propagated yet. Run manually: sudo certbot --nginx -d ${DNS_NAME}"

echo "SSL certificate installation attempted"

# Set up auto-renewal
systemctl enable certbot-renew.timer
systemctl start certbot-renew.timer

echo "=== CDR Mapper Instance Setup Completed at $(date) ==="
