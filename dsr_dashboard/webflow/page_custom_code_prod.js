<!-- First, load Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- Add the date-fns adapter for time-based charts -->
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

<!-- Third, add your chart containers -->
<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="era5_t2m_anom_year"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="aviso_slr"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="co2_ppm"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="home_insurance_premium"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="noaa_billion"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="era5_sst_anom_month_day"></canvas>
</div>

<div class="chart-container" style="width: 100%; max-width: 1200px; margin: 0 auto;">
    <canvas id="era5_t2m_anom_month_day"></canvas>
</div>

<script>
    const API_BASE_URL = 'https://xcp79dy519.execute-api.ca-central-1.amazonaws.com/prod';

    // Function to load chart templates dynamically
        async function loadChartTemplates() {
            try {
                console.log('Fetching chart templates...'); // Debug log
                const response = await fetch(`${API_BASE_URL}/presigned-url?key=viz/chart_templates.js`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log('Received presigned URL:', data.url); // Debug log
                
                // Create and load script
                const script = document.createElement('script');
                script.src = data.url;
                script.crossOrigin = "anonymous";
                
                // Create a promise to handle script loading
                const loadPromise = new Promise((resolve, reject) => {
                    script.onload = () => {
                        console.log('Chart templates loaded successfully');
                        resolve();
                    };
                    script.onerror = (error) => {
                        console.error('Error loading chart templates:', error);
                        reject(error);
                    };
                });

            // Add script to document
            document.head.appendChild(script);
            
            // Wait for script to load
            await loadPromise;
            
            // Initialize charts after successful load
            await initializeCharts();
            
        } catch (error) {
            console.error('Error in loadChartTemplates:', error);
        }
    }
    // Move chart initialization to separate function
    async function initializeCharts() {
        try {
            console.log('Initializing charts...'); // Debug log
            const charts = new DeepSkyCharts(API_BASE_URL);
            
            // Fix: Update the path for dataset configuration
            const configResponse = await fetch(`${API_BASE_URL}/presigned-url?key=dataset_dir.json`);
            if (!configResponse.ok) {
                throw new Error(`HTTP error! status: ${configResponse.status}`);
            }
            const configData = await configResponse.json();
            
            // Add error handling for the dataset config fetch
            const datasetConfigResponse = await fetch(configData.url);
            if (!datasetConfigResponse.ok) {
                throw new Error(`HTTP error! status: ${datasetConfigResponse.status}`);
            }
            const datasetConfig = await datasetConfigResponse.json();
            
            console.log('Dataset config loaded:', datasetConfig); // Debug log
            
            // Create chart configurations
            const chartConfigs = Object.entries(datasetConfig).map(([key, config]) => ({
                canvasId: key,
                type: config.chart_type,
                filename: `${key}_*.csv`,
                title: config.chart_title,
                yAxisLabel: config.y_axis_title,
                color: 'rgb(59, 130, 246)',
                x_var: config.x_axis_unit,
                y_var: config.chart_y_var || 'value'
            }));

            console.log('Chart configs created:', chartConfigs); // Debug log
            await charts.createCharts(chartConfigs);
        } catch (error) {
            console.error('Error in initializeCharts:', error);
        }
    }

    // Start loading process when DOM is ready
    document.addEventListener('DOMContentLoaded', loadChartTemplates);
</script>