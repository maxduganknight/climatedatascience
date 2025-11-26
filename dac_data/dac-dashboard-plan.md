# Direct Air Capture Dashboard Technical Plan

## Project Overview
Build a live-appearing public dashboard for Direct Air Capture (DAC) carbon removal metrics, displaying environmental conditions, CO2 flow rates, removal efficiency, DAC machine status, tonnes removed to date with smooth animations despite hourly data updates.

## Architecture Summary
**Data Flow**: Gold Layer (S3) → Databricks (Python/SQL) → Public Dashboard S3 (CSV) → Webflow Website (JavaScript)

## Technical Stack
- **Data Processing**: Databricks with Python/PySpark
- **Storage**: AWS S3
- **Data Format**: CSV files
- **Frontend**: JavaScript
- **Hosting**: Webflow with custom JavaScript
- **Update Frequency**: Hourly

## Detailed Architecture

### 1. Data Pipeline Architecture

#### Source Data
- **Location**: S3 Gold Layer (Parquet format)
- **Access**: Via Databricks using PySpark
- **Update Frequency**: Hourly

#### Processing Layer
- **Platform**: Databricks
- **Language**: Python (PySpark/Pandas)
- **Job Type**: Scheduled notebook (hourly)
- **Key Transformations**:
  - Read from Gold layer parquet files
  - Calculate current metrics
  - Generate historical trends (48-hour rolling window)
  - Compute daily statistics (24h average, min, max)
  - Format data for dashboard consumption

#### Output Data
- **Location**: S3 bucket (`public-dac-dashboard`)
- **Format**: CSV files
- **Access Method**: Presigned URLs

### 2. S3 Bucket Structure

```
public-dac-dashboard/
├── data/
│   ├── current_snapshot_*.csv    # Latest values for all metrics
│   ├── hourly_trends_*.csv       # 48-hour rolling window
│   └── metadata.csv              # Update timestamps and data quality
├── js/
│   ├── dac_dashboard.js          # Main dashboard application
├── config/
│   └── dac_dashboard_config.json # Data configuration
└── logs/
    ├── databricks/               # Pipeline execution logs
    │   └── pipeline_*.log
    ├── lambda/                   # Lambda function logs
    │   └── access_*.log
    └── frontend/                 # Client-side error logs
        └── errors_*.log
```

Note: CSV files use wildcard pattern with timestamps (e.g., current_snapshot_20250625.csv) to support versioning and the Lambda presigned URL pattern matching.

### 3. Data Schemas

#### current_snapshot.csv
```csv
timestamp,metric,value,unit
2025-06-25T09:35:00Z,temperature,16.3,celsius
2025-06-25T09:35:00Z,humidity,68,percent
2025-06-25T09:35:00Z,air_quality,24,AQI
2025-06-25T09:35:00Z,ambient_co2,427,ppm
2025-06-25T09:35:00Z,liquefaction_flow,163.5,kg/h
2025-06-25T09:35:00Z,tank_flow,64.7,kg/h
2025-06-25T09:35:00Z,storage_flow,0,kg/h
2025-06-25T09:35:00Z,removal_efficiency,77.3,percent
2025-06-25T09:35:00Z,tonnes_removed_total,12.18765678,tonnes
```

#### hourly_trends.csv
```csv
timestamp,metric,start_value,end_value,hourly_min,hourly_max,hourly_mean
2025-06-25T09:00:00Z,liquefaction_flow,161.2,163.5,0,188.3,140.6
2025-06-25T09:00:00Z,tank_flow,70.3,64.7,0,110.5,72.5
2025-06-25T09:00:00Z,storage_flow,0,0,0,786.2,12.3
2025-06-25T09:00:00Z,removal_efficiency,74.0,77.3,66.8,83.5,78.2
2025-06-25T09:00:00Z,tonnes_removed_total,12.18765673,12.18765678,12.18765673,12.18765678,12.18765675
```

#### metadata.csv
```csv
key,value,timestamp
last_update,success,2025-06-25T09:35:00Z
data_quality,good,2025-06-25T09:35:00Z
pipeline_version,1.0.0,2025-06-25T09:35:00Z
```

### 4. Databricks Pipeline Implementation

#### Main Processing Notebook Structure
```python
# 1. Configuration
OUTPUT_BUCKET = "public-dac-dashboard"
GOLD_LAYER_PATH = "s3://deepsky-gold-layer-data-prod/"

# 2. Read from Gold Layer
def read_gold_data():
    # Read parquet files from Gold layer
    # Return DataFrames for each data type
    pass

# 3. Transform Data
def calculate_current_metrics(df):
    # Extract latest values
    # Format for current_snapshot.csv
    pass

def generate_hourly_trends(df, hours=48):
    # Create rolling window of hourly data
    # Format for hourly_trends.csv
    pass

def calculate_daily_summary(df):
    # Aggregate daily statistics
    # Format for daily_summary.csv
    pass

# 4. Write to Public S3
def write_csv_to_s3(df, bucket, key_prefix):
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d')
    key = f"{key_prefix}_{timestamp}.csv"
    
    # Convert DataFrame to CSV
    csv_buffer = df.to_csv(index=False)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer,
        ContentType='text/csv'
    )
    
    # Clean up old versions (keep last 7 days)
    cleanup_old_files(bucket, key_prefix, days_to_keep=7)

# 5. Main Execution
def main():
    # Read data
    gold_data = read_gold_data()
    
    # Transform
    current_metrics = calculate_current_metrics(gold_data)
    hourly_trends = generate_hourly_trends(gold_data)
    
    # Write outputs with timestamps
    write_csv_to_s3(current_metrics, OUTPUT_BUCKET, "data/current_snapshot")
    write_csv_to_s3(hourly_trends, OUTPUT_BUCKET, "data/hourly_trends")
    
    # Update metadata
    update_metadata(OUTPUT_BUCKET)
```

### 5. Frontend Implementation

### Lambda Function for Presigned URLs
```python
import os
import json
import boto3
import re
import logging
from botocore.exceptions import ClientError
from datetime import datetime
from functools import wraps

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Rate limiting decorator
def rate_limit(max_calls=100, window=60):
    calls = {}
    def decorator(func):
        @wraps(func)
        def wrapper(event, context):
            # Get client IP
            source_ip = event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')
            current_time = datetime.now().timestamp()
            
            # Clean old entries
            calls[source_ip] = [t for t in calls.get(source_ip, []) if current_time - t < window]
            
            # Check rate limit
            if len(calls.get(source_ip, [])) >= max_calls:
                logger.warning(f"Rate limit exceeded for IP: {source_ip}")
                return {
                    'statusCode': 429,
                    'headers': get_cors_headers(event),
                    'body': json.dumps({'error': 'Too many requests'})
                }
            
            # Record this call
            calls.setdefault(source_ip, []).append(current_time)
            return func(event, context)
        return wrapper
    return decorator

def validate_key(key):
    """Validate requested key to prevent path traversal attacks"""
    # Whitelist allowed prefixes
    allowed_prefixes = ['data/', 'js/', 'config/']
    
    # Check for path traversal attempts
    if '..' in key or key.startswith('/') or '\\' in key:
        raise ValueError("Invalid key format - potential path traversal")
    
    # Ensure key starts with allowed prefix
    if not any(key.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(f"Key must start with one of: {allowed_prefixes}")
    
    # Validate file extensions
    allowed_extensions = ['.csv', '.js', '.json']
    if not any(key.endswith(ext) for ext in allowed_extensions):
        raise ValueError(f"File must have one of these extensions: {allowed_extensions}")
    
    return key

def get_cors_headers(event):
    """Get CORS headers based on request origin"""
    allowed_origins = [
        'https://www.deepskyclimate.com',
        'https://deepsky.webflow.io'
    ]
    
    origin = event.get('headers', {}).get('origin', 'https://www.deepskyclimate.com')
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': origin if origin in allowed_origins else allowed_origins[0],
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type,Origin'
    }

def log_access(event, status_code, key=None):
    """Log access attempts to S3"""
    try:
        source_ip = event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')
        timestamp = datetime.utcnow().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'source_ip': source_ip,
            'requested_key': key,
            'status_code': status_code,
            'user_agent': event.get('headers', {}).get('User-Agent', 'unknown')
        }
        
        # Write to CloudWatch (automatically captured by Lambda)
        logger.info(json.dumps(log_entry))
        
        # Also write to S3 for long-term storage (async)
        # This could be done via CloudWatch Logs export to S3
        
    except Exception as e:
        logger.error(f"Failed to log access: {str(e)}")

@rate_limit(max_calls=100, window=60)
def handler(event, context):
    headers = get_cors_headers(event)
    
    try:
        s3 = boto3.client('s3')
        bucket_name = os.environ.get('DAC_DASHBOARD_BUCKET', 'public-dac-dashboard')
        requested_key = event.get('queryStringParameters', {}).get('key')
        
        if not requested_key:
            log_access(event, 400)
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Missing key parameter'})
            }
        
        # Validate the requested key
        try:
            validated_key = validate_key(requested_key)
        except ValueError as e:
            log_access(event, 400, requested_key)
            logger.warning(f"Invalid key requested: {requested_key} - {str(e)}")
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Invalid key format'})
            }
        
        # Handle wildcard patterns for latest file
        if '*' in validated_key:
            prefix = validated_key.split('*')[0]
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            # Find most recent file
            latest_file = None
            latest_date = None
            
            for obj in response.get('Contents', []):
                match = re.search(r'(\d{8})', obj['Key'])
                if match:
                    file_date = datetime.strptime(match.group(1), '%Y%m%d')
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = obj['Key']
            
            if not latest_file:
                log_access(event, 404, validated_key)
                return {
                    'statusCode': 404,
                    'headers': headers,
                    'body': json.dumps({'error': 'No matching files found'})
                }
            
            validated_key = latest_file
        
        # Generate presigned URL
        content_type = 'application/javascript' if validated_key.endswith('.js') else 'text/csv'
        if validated_key.endswith('.json'):
            content_type = 'application/json'
            
        signed_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': validated_key,
                'ResponseContentType': content_type
            },
            ExpiresIn=300  # 5 minutes
        )
        
        log_access(event, 200, validated_key)
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'url': signed_url})
        }
    
    except ClientError as e:
        log_access(event, 500, requested_key)
        logger.error(f"S3 error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Internal server error'})
        }
    
    except Exception as e:
        log_access(event, 500, requested_key)
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Internal server error'})
        }
```

### JavaScript Dashboard Class
```javascript
class DACDashboard {
    constructor(apiBaseUrl, options = {}) {
        this.API_BASE_URL = apiBaseUrl;
        this.charts = new Map();
        this.latestValues = new Map();
        this.cache = new Map();
        this.cacheExpiry = new Map();
        this.CACHE_DURATION = 1000 * 60 * 60; // 1 hour cache
        this.config = null;
        this.errorState = false;
        
        // Performance metrics for monitoring
        this.metrics = {
            loadTimes: [],
            errors: [],
            cacheHits: 0,
            cacheMisses: 0
        };
    }
    
    async loadConfig() {
        try {
            const configData = await this.fetchJSONData('config/dac_dashboard_config.json');
            this.config = configData;
            
            // Set up formatters based on config
            this.formatters = {};
            Object.entries(this.config).forEach(([metric, config]) => {
                if (config.formatter === 'decimal') {
                    this.formatters[metric] = value => value.toFixed(config.decimal_places || 1);
                } else if (config.formatter === 'integer') {
                    this.formatters[metric] = value => Math.round(value).toString();
                }
            });
        } catch (error) {
            console.error('Failed to load configuration:', error);
            throw new Error('Configuration load failed');
        }
    }
    
    validateCSVData(data, expectedFields) {
        // Validate data structure
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Invalid data format: expected non-empty array');
        }
        
        // Check required fields
        const requiredFields = expectedFields || ['timestamp', 'metric', 'value', 'unit'];
        const hasRequiredFields = data.every(row => 
            requiredFields.every(field => field in row && row[field] !== null)
        );
        
        if (!hasRequiredFields) {
            throw new Error('Missing required fields in data');
        }
        
        // Validate data types
        data.forEach((row, index) => {
            if (typeof row.value !== 'number') {
                throw new Error(`Invalid value type at row ${index}: expected number`);
            }
            if (row.timestamp && isNaN(Date.parse(row.timestamp))) {
                throw new Error(`Invalid timestamp at row ${index}`);
            }
        });
        
        return true;
    }
    
    async fetchJSONData(filename) {
        const url = await this.getPresignedUrl(filename);
        const response = await fetch(url);
        return await response.json();
    }
    
    async fetchCSVData(filename, retries = 3) {
        const startTime = Date.now();
        
        for (let i = 0; i < retries; i++) {
            try {
                const url = await this.getPresignedUrl(filename);
                const response = await fetch(url);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const csvText = await response.text();
                const parsed = Papa.parse(csvText, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    transformHeader: header => header.trim() // Strip whitespace from headers
                });
                
                if (parsed.errors.length > 0) {
                    console.warn('CSV parsing warnings:', parsed.errors);
                }
                
                // Validate the parsed data
                this.validateCSVData(parsed.data);
                
                // Record successful load time
                this.metrics.loadTimes.push(Date.now() - startTime);
                
                return parsed.data;
                
            } catch (error) {
                console.warn(`Attempt ${i + 1} failed:`, error);
                this.metrics.errors.push({
                    timestamp: new Date().toISOString(),
                    error: error.message,
                    filename: filename
                });
                
                if (i === retries - 1) {
                    // Try to use cached data as fallback
                    const cached = this.getFromCache(filename);
                    if (cached) {
                        console.log('Using cached data as fallback');
                        return cached;
                    }
                    throw error;
                }
                
                // Exponential backoff
                await new Promise(r => setTimeout(r, Math.pow(2, i) * 1000));
            }
        }
    }
    
    showErrorState(message = 'Unable to load data. Please try again later.') {
        this.errorState = true;
        
        // Update all metric displays to show error state
        document.querySelectorAll('.metric-value').forEach(el => {
            el.textContent = '--';
            el.classList.add('error');
        });
        
        // Show error message
        const errorEl = document.getElementById('error-message');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = 'block';
        }
        
        // Log error to backend
        this.logError({
            type: 'data_load_failure',
            message: message,
            timestamp: new Date().toISOString()
        });
    }
    
    async logError(errorData) {
        try {
            // Send error logs to backend
            // This could be implemented as another Lambda function
            await fetch(`${this.API_BASE_URL}/log-error`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': window.location.origin
                },
                body: JSON.stringify({
                    ...errorData,
                    userAgent: navigator.userAgent,
                    url: window.location.href
                })
            });
        } catch (e) {
            console.error('Failed to log error:', e);
        }
    }
    
    startAnimations() {
        // Animate values between updates based on config
        this.animationInterval = setInterval(() => {
            this.latestValues.forEach((baseValue, metric) => {
                const element = document.getElementById(`${metric}_value`);
                const metricConfig = this.config[metric];
                
                if (element && typeof baseValue === 'number' && metricConfig) {
                    let animatedValue = baseValue;
                    
                    if (metricConfig.bidirectional === true) {
                        // Bidirectional animation (can go up or down)
                        const variation = (Math.random() - 0.5) * 0.04 * baseValue;
                        animatedValue = baseValue + variation;
                    } else if (metricConfig.bidirectional === false) {
                        // Unidirectional animation (only increases)
                        const variation = Math.random() * 0.02 * baseValue;
                        animatedValue = baseValue + variation;
                    }
                    
                    const formatter = this.formatters[metric];
                    element.textContent = formatter ? 
                        formatter(animatedValue) : animatedValue.toFixed(1);
                }
            });
        }, 3000); // Update every 3 seconds
    }
    
    async updateMetrics() {
        try {
            // Fetch current snapshot
            const currentData = await this.fetchCSVData('data/current_snapshot_*.csv');
            
            // Clear any previous error state
            if (this.errorState) {
                this.errorState = false;
                document.querySelectorAll('.metric-value.error').forEach(el => {
                    el.classList.remove('error');
                });
                const errorEl = document.getElementById('error-message');
                if (errorEl) errorEl.style.display = 'none';
            }
            
            // Update display values
            currentData.forEach(row => {
                const element = document.getElementById(`${row.metric}_value`);
                const metricConfig = this.config[row.metric];
                
                if (element && metricConfig) {
                    const formatter = this.formatters[row.metric];
                    const value = formatter ? formatter(row.value) : row.value;
                    element.textContent = value;
                    
                    // Update unit if needed
                    const unitElement = document.getElementById(`${row.metric}_unit`);
                    if (unitElement && metricConfig.display_unit) {
                        unitElement.textContent = metricConfig.display_unit;
                    }
                    
                    // Store for animations
                    this.latestValues.set(row.metric, row.value);
                }
            });
            
            // Update timestamp and data quality indicator
            const timestampEl = document.getElementById('last_update');
            if (timestampEl && currentData[0]) {
                const date = new Date(currentData[0].timestamp);
                timestampEl.textContent = date.toLocaleString();
                this.updateDataQuality(currentData[0].timestamp);
            }
            
        } catch (error) {
            console.error('Error updating metrics:', error);
            this.showErrorState();
        }
    }
    
    updateDataQuality(timestamp) {
        const age = Date.now() - new Date(timestamp).getTime();
        const hoursOld = age / (1000 * 60 * 60);
        
        const indicator = document.getElementById('data-quality-indicator');
        if (indicator) {
            if (hoursOld < 1) {
                indicator.className = 'data-quality fresh'; // Green
                indicator.title = 'Data is current';
            } else if (hoursOld < 2) {
                indicator.className = 'data-quality aging'; // Yellow
                indicator.title = 'Data is slightly outdated';
            } else {
                indicator.className = 'data-quality stale'; // Red
                indicator.title = 'Data is stale - check pipeline';
            }
        }
    }
    
    async initialize() {
        try {
            // Load configuration first
            await this.loadConfig();
            
            // Initial data load
            await this.updateMetrics();
            
            // Start animations
            this.startAnimations();
            
            // Schedule hourly updates
            setInterval(() => this.updateMetrics(), 3600000);
            
            // Report metrics every 5 minutes
            setInterval(() => this.reportMetrics(), 300000);
            
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showErrorState('Failed to initialize dashboard');
        }
    }
    
    async reportMetrics() {
        // Calculate and log performance metrics
        if (this.metrics.loadTimes.length > 0) {
            const avgLoadTime = this.metrics.loadTimes.reduce((a,b) => a+b, 0) / this.metrics.loadTimes.length;
            const cacheHitRate = this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses) || 0;
            
            console.log('Dashboard metrics:', {
                avgLoadTime: Math.round(avgLoadTime) + 'ms',
                errorCount: this.metrics.errors.length,
                cacheHitRate: (cacheHitRate * 100).toFixed(1) + '%',
                totalLoads: this.metrics.loadTimes.length
            });
            
            // Send metrics to backend if needed
            await this.logError({
                type: 'performance_metrics',
                metrics: {
                    avgLoadTime,
                    errorCount: this.metrics.errors.length,
                    cacheHitRate
                }
            });
            
            // Reset metrics
            this.metrics.loadTimes = [];
            this.metrics.errors = [];
        }
    }
}
```

### Webflow Integration
In your Webflow site, add the following to the page where the dashboard will be displayed:

```html
<!-- In the <head> section -->
<script src="https://cdn.jsdelivr.net/npm/papaparse@5.3.0/papaparse.min.js"></script>

<!-- Before closing </body> tag -->
<script>
  // Load the dashboard script from S3
  const script = document.createElement('script');
  script.src = 'YOUR_API_GATEWAY_URL/presigned-url?key=js/dac_dashboard.js';
  script.onload = function() {
    // Initialize dashboard after script loads
    const dashboard = new DACDashboard('YOUR_API_GATEWAY_URL');
    dashboard.initialize();
  };
  document.body.appendChild(script);
</script>
```

#### HTML Structure
Create div elements in Webflow with IDs matching your metrics:
```html
<div class="metric-container">
  <h3>Temperature</h3>
  <div id="temperature_value" class="metric-value">--</div>
  <div class="metric-unit">°C</div>
</div>

<div class="metric-container">
  <h3>CO2 Capture Flow</h3>
  <div id="capture_flow_value" class="metric-value">--</div>
  <div class="metric-unit">kg/h</div>
</div>

<!-- Last update timestamp -->
<div id="last_update" class="update-time">Last updated: --</div>
```

### 6. Security Considerations

1. **Lambda Function Setup**
   - Deploy Lambda function for generating presigned URLs
   - Configure environment variables:
     - `DAC_DASHBOARD_BUCKET`: S3 bucket name
   - Set appropriate IAM role with S3 read permissions
   - Configure API Gateway trigger

2. **S3 Bucket Configuration**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "AWS": "arn:aws:iam::ACCOUNT_ID:role/LambdaPresignedUrlRole"
         },
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::public-dac-dashboard/*"
       }
     ]
   }
   ```

3. **CORS Configuration**
   - Configure allowed origins in Lambda function
   - Include production domain and test environments
   - Handle preflight OPTIONS requests

4. **API Gateway Setup**
   - Create REST API with GET method
   - Enable CORS
   - Deploy to prod stage
   - Note the invoke URL for frontend configuration

### 7. Logging Architecture

#### Databricks Logging
Databricks provides built-in logging capabilities that we'll enhance with custom logging:

1. **Built-in Databricks Features**:
   - Job run history and logs automatically stored
   - Cluster logs available in driver logs
   - Notebook execution history tracked
   - Integration with Azure Monitor/AWS CloudWatch

2. **Custom Logging Implementation**:
```python
import logging
import json
from datetime import datetime
import boto3

class DACPipelineLogger:
    def __init__(self, bucket_name, log_prefix="logs/databricks"):
        self.bucket_name = bucket_name
        self.log_prefix = log_prefix
        self.s3_client = boto3.client('s3')
        self.logs = []
        
        # Configure local logger
        self.logger = logging.getLogger('DACPipeline')
        self.logger.setLevel(logging.INFO)
        
        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log(self, level, message, **kwargs):
        """Log message locally and queue for S3 upload"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'metadata': kwargs
        }
        
        self.logs.append(log_entry)
        getattr(self.logger, level.lower())(f"{message} - {json.dumps(kwargs)}")
    
    def write_to_s3(self):
        """Write accumulated logs to S3"""
        if not self.logs:
            return
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        key = f"{self.log_prefix}/pipeline_{timestamp}.log"
        
        log_content = '\n'.join(json.dumps(log) for log in self.logs)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=log_content,
            ContentType='application/json'
        )
        
        self.logger.info(f"Wrote {len(self.logs)} log entries to s3://{self.bucket_name}/{key}")
        self.logs = []  # Clear after writing

# Usage in pipeline
logger = DACPipelineLogger('public-dac-dashboard')

def main():
    logger.log('INFO', 'Pipeline started', job_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().jobId().get())
    
    try:
        # Read data
        logger.log('INFO', 'Reading from Gold layer', path=GOLD_LAYER_PATH)
        gold_data = read_gold_data()
        logger.log('INFO', 'Successfully read Gold data', row_count=len(gold_data))
        
        # Transform
        logger.log('INFO', 'Calculating current metrics')
        current_metrics = calculate_current_metrics(gold_data)
        logger.log('INFO', 'Generated current metrics', metric_count=len(current_metrics))
        
        # Write outputs
        logger.log('INFO', 'Writing to S3', bucket=OUTPUT_BUCKET)
        write_csv_to_s3(current_metrics, OUTPUT_BUCKET, "data/current_snapshot")
        
        logger.log('INFO', 'Pipeline completed successfully')
        
    except Exception as e:
        logger.log('ERROR', 'Pipeline failed', error=str(e), traceback=traceback.format_exc())
        raise
    
    finally:
        # Always write logs to S3
        logger.write_to_s3()
```
#### Lambda Logging

   - CloudWatch Logs automatically capture Lambda stdout/stderr
   - Custom access logs written to S3 for long-term analysis
   - Structured logging with JSON format for easy querying

#### Frontend Error Logging
```javascript
window.addEventListener('error', function(event) {
    dashboard.logError({
        type: 'javascript_error',
        message: event.message,
        filename: event.filename,
        line: event.lineno,
        column: event.colno,
        stack: event.error?.stack
    });
});
```

#### Log Retention and Analysis

   - S3 Lifecycle policies: 90 days for debug logs, 1 year for error logs
   - CloudWatch Insights for real-time analysis
   - Athena for historical log analysis


### 8. Monitoring & Maintenance

#### Key Metrics to Track
- Pipeline execution time
- Data freshness (time since last update)
- File sizes
- Frontend load times
- Error rates

#### Alerting Rules
- Pipeline failure
- Data staleness > 2 hours
- File size > 1MB
- High error rates in frontend

### 9. Configuration Management

#### Dashboard Configuration Structure
The dashboard behavior is controlled by `dac_dashboard_config.json`, which defines display properties, animation behavior, and data handling for each metric.

Configuration is loaded at dashboard initialization and controls:
- Display formatting (decimal places, units)
- Animation behavior (bidirectional vs unidirectional)
- Update thresholds and validation rules
- Visual styling preferences

Example configuration entry:
```json
"temperature": {
    "display_name": "Temperature",
    "display_unit": "°C",
    "data_unit": "celsius",
    "formatter": "decimal",
    "decimal_places": 1,
    "bidirectional": true,
    "min_valid_value": -50,
    "max_valid_value": 60,
    "update_threshold": 0.1
}
```

### 10. Development Guidelines

#### Python/Databricks Best Practices
- Use PySpark for large datasets
- Implement proper error handling
- Add comprehensive logging
- Write unit tests for transformations
- Use configuration files for parameters

#### JavaScript Best Practices
- Implement graceful degradation
- Add loading states
- Cache data appropriately
- Use async/await for data loading
- Minimize bundle size

#### CSV Optimization
- Keep files under 1MB
- Use appropriate precision for numbers
- Consider gzip compression for larger files
- Implement incremental updates where possible

### 11. Testing

Testing Architecture
tests/
├── unit/
│   ├── test_data_transformations.py    # PySpark transformation logic
│   ├── test_metric_calculations.py     # Metric calculation functions
│   ├── test_data_validators.py         # Data validation logic
│   └── test_config_parser.py           # Configuration parsing
├── integration/
│   ├── test_databricks_pipeline.py     # Full pipeline tests
│   ├── test_s3_operations.py           # S3 read/write operations
│   ├── test_lambda_functions.py        # Lambda function tests
│   └── test_api_endpoints.py           # API Gateway integration
├── frontend/
│   ├── test_dashboard_class.py         # JavaScript unit tests
│   ├── test_data_parsing.py            # CSV/JSON parsing tests
│   ├── test_animations.py              # Animation logic tests
│   └── test_error_handling.py          # Frontend error scenarios
├── e2e/
│   ├── test_full_pipeline.py           # Complete data flow
│   ├── test_data_freshness.py          # Data timeliness checks
│   ├── test_dashboard_display.py       # Visual regression tests
│   └── test_performance.py             # Load time and responsiveness
├── databricks/
│   ├── test_notebooks/                  # Test notebooks for Databricks
│   │   ├── test_gold_layer_read.py
│   │   └── test_csv_generation.py
│   └── test_job_configs/                # Job configuration tests
├── fixtures/
│   ├── sample_gold_data.parquet        # Mock Gold layer data
│   ├── expected_outputs/                # Expected CSV outputs
│   └── config_samples/                  # Test configurations
└── conftest.py                          # Shared pytest fixtures

#### Databricks Testing Approach

```python
# tests/unit/test_data_transformations.py
import pytest
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from pipeline.transformations import calculate_current_metrics

@pytest.fixture(scope="session")
def spark():
    """Create Databricks Connect session for testing"""
    return DatabricksSession.builder.remote(
        host="https://your-workspace.databricks.com",
        token="your-token",
        cluster_id="your-cluster-id"
    ).getOrCreate()

def test_calculate_current_metrics(spark, sample_gold_data):
    """Test current metrics calculation"""
    df = spark.read.parquet(sample_gold_data)
    result = calculate_current_metrics(df)
    
    assert result.count() == 9  # Expected number of metrics
    assert 'temperature' in result.select('metric').rdd.flatMap(lambda x: x).collect()
    assert result.filter("metric = 'temperature'").select('value').first()[0] is not None
```

#### Lambda Testing Approach
```python
# tests/integration/test_lambda_functions.py
import pytest
import json
from unittest.mock import patch, MagicMock
from lambda_functions.presigned_url import handler, validate_key, rate_limit

class TestPresignedUrlLambda:
    def test_validate_key_valid_paths(self):
        """Test key validation accepts valid paths"""
        valid_keys = [
            'data/current_snapshot_20250625.csv',
            'js/dac_dashboard.js',
            'config/dac_dashboard_config.json'
        ]
        for key in valid_keys:
            assert validate_key(key) == key
    
    def test_validate_key_path_traversal(self):
        """Test key validation prevents path traversal"""
        malicious_keys = [
            '../../../etc/passwd',
            'data/../../private/secrets.csv',
            '/absolute/path/file.csv',
            'data\\..\\..\\windows\\system32'
        ]
        for key in malicious_keys:
            with pytest.raises(ValueError, match="Invalid key format"):
                validate_key(key)
    
    @patch('lambda_functions.presigned_url.datetime')
    def test_rate_limiting(self, mock_datetime):
        """Test rate limiting functionality"""
        mock_datetime.now.return_value.timestamp.return_value = 1000
        
        # Create a simple function to test
        @rate_limit(max_calls=2, window=60)
        def test_func(event, context):
            return {'statusCode': 200}
        
        event = {
            'requestContext': {'identity': {'sourceIp': '1.2.3.4'}},
            'headers': {}
        }
        
        # First two calls should succeed
        assert test_func(event, {})['statusCode'] == 200
        assert test_func(event, {})['statusCode'] == 200
        
        # Third call should be rate limited
        result = test_func(event, {})
        assert result['statusCode'] == 429
        assert 'Too many requests' in result['body']
```

#### Frontend Testing
```python
# tests/frontend/test_dashboard_class.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestDACDashboard:
    @pytest.fixture
    def mock_api_server(self):
        """Create a mock API server for testing"""
        # Use pytest-httpserver or similar
        pass
    
    @pytest.fixture
    def test_page(self, tmp_path, mock_api_server):
        """Create test HTML page with dashboard"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/papaparse@5.3.0/papaparse.min.js"></script>
        </head>
        <body>
            <div id="temperature_value" class="metric-value">--</div>
            <div id="humidity_value" class="metric-value">--</div>
            <div id="tonnes_removed_total_value" class="metric-value">--</div>
            <div id="error-message" style="display:none;"></div>
            <div id="data-quality-indicator"></div>
            <script>
                // Dashboard code here
            </script>
        </body>
        </html>
        """
        # Write and return path
    
    def test_bidirectional_animation(self, browser, test_page):
        """Test bidirectional vs unidirectional animations"""
        browser.get(f"file://{test_page}")
        
        # Wait for dashboard to load
        WebDriverWait(browser, 10).until(
            EC.text_to_be_present_in_element((By.ID, "temperature_value"), "16.3")
        )
        
        # Capture initial values
        initial_temp = float(browser.find_element(By.ID, "temperature_value").text)
        initial_tonnes = float(browser.find_element(By.ID, "tonnes_removed_total_value").text)
        
        # Wait for animation cycle
        browser.implicitly_wait(4)
        
        # Check temperature can go up or down (bidirectional)
        temp_values = []
        for _ in range(5):
            temp_values.append(float(browser.find_element(By.ID, "temperature_value").text))
            browser.implicitly_wait(3)
        
        assert any(v < initial_temp for v in temp_values), "Temperature should decrease sometimes"
        assert any(v > initial_temp for v in temp_values), "Temperature should increase sometimes"
        
        # Check tonnes only increases (unidirectional)
        tonnes_values = []
        for _ in range(5):
            tonnes_values.append(float(browser.find_element(By.ID, "tonnes_removed_total_value").text))
            browser.implicitly_wait(3)
        
        assert all(v >= initial_tonnes for v in tonnes_values), "Tonnes should never decrease"
```

#### End-to-end Testing
```python
# tests/e2e/test_full_pipeline.py
import pytest
import time
import pandas as pd
from datetime import datetime, timedelta

class TestFullPipeline:
    @pytest.fixture
    def trigger_pipeline(self):
        """Fixture to trigger Databricks job"""
        # Use Databricks REST API to trigger job
        pass
    
    def test_data_freshness(self, trigger_pipeline):
        """Test that data is updated within expected timeframe"""
        # Trigger pipeline
        job_run_id = trigger_pipeline()
        
        # Wait for completion (with timeout)
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check job status
            if job_is_complete(job_run_id):
                break
            time.sleep(10)
        
        # Verify output files exist and are recent
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket='public-dac-dashboard',
            Prefix='data/current_snapshot_'
        )
        
        latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
        file_age = datetime.now(timezone.utc) - latest_file['LastModified']
        
        assert file_age < timedelta(minutes=10), "Data file is too old"
    
    def test_data_validation_rules(self):
        """Test that all validation rules from config are enforced"""
        # Load config
        with open('config/dac_dashboard_config.json') as f:
            config = json.load(f)
        
        # Read latest data
        df = pd.read_csv('s3://public-dac-dashboard/data/current_snapshot_*.csv')
        
        # Validate each metric against config
        for metric, rules in config.items():
            metric_data = df[df['metric'] == metric]
            
            if 'min_valid_value' in rules:
                assert all(metric_data['value'] >= rules['min_valid_value'])
            
            if 'max_valid_value' in rules:
                assert all(metric_data['value'] <= rules['max_valid_value'])
```

## Contact & Resources

- **Architecture Diagrams**: See attached images
- **Databricks Documentation**: [docs.databricks.com](https://docs.databricks.com)
- **AWS S3 Presigned URLs**: [AWS Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)

---

*This document serves as the primary reference for all team members and future Claude models working on the Direct Air Capture Dashboard project.*