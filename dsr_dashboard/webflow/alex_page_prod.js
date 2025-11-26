<script>

document.addEventListener("DOMContentLoaded", ()=> {
var wLocation = window.location.href;
console.log("This is the url", wLocation)
if (wLocation.includes("https://fr.deepskyclimate.com") == true) {
	console.log("This is french")
	document.getElementById("french-lang").style.display = "none";
} else if (wLocation.includes("https://www.deepskyclimate.com") == true) {
	console.log("This is english")
	document.getElementById("english-lang").style.display = "none";
}


})


$('.w-nav-menu').on('click', 'a', function() {

  // When a nav item is clicked on a tablet or mobile device

    if (parseInt($(window).width()) < 990) {

	// Click the menu close button

        $('.w-nav-button').triggerHandler('tap');

    }

});
</script>

<link rel="preload" href="https://fonts.gstatic.com/s/spacemono/v13/i7dPIFZifjKcF5UAWdDRYE98RXi4EwSsbg.woff2" as="font" type="font/woff2" crossorigin>
<!-- Add font-display: swap to ensure text remains visible during font loading -->
<link href="https://fonts.googleapis.com/css2?family=Space+Mono&display=swap&text=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,%°$-" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- Add the date-fns adapter for time-based charts -->
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

<script>
    const API_BASE_URL = 'https://xcp79dy519.execute-api.ca-central-1.amazonaws.com/prod';

    let chartsInstance; // Add this to store the charts instance

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
            chartsInstance = new DeepSkyCharts(API_BASE_URL);
            
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
                y_axis_unit: config.y_axis_unit,
                color: 'rgb(59, 130, 246)',
                x_var: config.x_axis_unit,
                y_var: config.chart_y_var || 'value'
            }));

            console.log('Chart configs created:', chartConfigs); // Debug log
            
            // Use the progressive loading strategy
            await chartsInstance.createChartsProgressively(chartConfigs);
            
            // No need to call updateDisplayValues here anymore
            // as it's handled inside createChartsProgressively
            
            // Schedule periodic checks for updates (once every 4 hours)
            setInterval(async () => {
                const hasUpdates = await chartsInstance.checkForNewVersions();
                if (hasUpdates) {
                    console.log('Found new data versions, refreshing charts...');
                    window.location.reload();
                }
            }, 4 * 60 * 60 * 1000);
            
        } catch (error) {
            console.error('Error in initializeCharts:', error);
        }
    }

    // Add this after initializing charts
    let lastUpdateCheck = Date.now();
    const UPDATE_CHECK_INTERVAL = 4 * 60 * 60 * 1000; // 4 hours

    // Check for updates when page becomes visible
    document.addEventListener('visibilitychange', async () => {
        if (document.visibilityState === 'visible') {
            const now = Date.now();
            if (now - lastUpdateCheck > UPDATE_CHECK_INTERVAL) {
                console.log('Page visible, checking for updates...');
                lastUpdateCheck = now;
                // Force a version check
                chartsInstance.forceVersionCheck = true;
                const hasUpdates = await chartsInstance.checkForNewVersions();
                if (hasUpdates) {
                    console.log('Found new data versions, refreshing charts...');
                    window.location.reload();
                }
            }
        }
    });
    
    // Start loading process when DOM is ready
    document.addEventListener('DOMContentLoaded', loadChartTemplates);
</script>
