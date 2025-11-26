<!-- CDR Chart Webflow Integration -->
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preload" href="https://fonts.gstatic.com/s/spacemono/v13/i7dPIFZifjKcF5UAWdDRYE98RXi4EwSsbg.woff2" as="font" type="font/woff2" crossorigin="anonymous">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono&display=swap&text=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,%°$-" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">

<!-- Chart.js libraries -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

<style>

.deals-ticker-container {   
    margin: 0px auto !important;
    }
    
    /* Main container for consistent alignment */
    .dashboard-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 20px;
        position: relative; /* For absolute positioning of stats */
    }
    
    .chart-container {
        position: relative;
        width: 100%;
        margin-bottom: 0px;
        height: 430px;
        clear: both; /* Ensure it clears the float */
    }
    
    .chart-title {
        color: #333333;
        font-size: 24px;
        margin-bottom: 0px;
        margin-top: 0px;
        text-align: left;
        text-transform: uppercase;
        font-family: 'Space Mono', monospace;
        float: left; /* Float left to allow stats to align */
    }
    
    #status-display {
        color: #666666;
        margin-top: 10px;
        font-size: 14px;
        text-align: center;
        clear: both; /* Ensure it appears below the chart */
    }
    
    .error-message {
        color: #DC3545;
        text-align: center;
        padding: 20px;
    }
    
    .stats-container {
        float: right; /* Float right to align with title */
        margin-bottom: 20px;
        margin-top: 00px; /* Match the chart-title margin-top */
        display: flex; /* Add flex display to arrange metrics side by side */
    }
    
    .stat-box {
        text-align: right;
        padding: 15px 20px;
        min-width: 180px; /* Adjusted width to fit both boxes */
    }
    
    .stat-label {
        color: #666666;
        font-size: 15px;
        text-transform: none;
        text-align: right;
        margin-top: 0px;
    }
    
    .delivery-percentage {
        color: #20DD7B;
        font-size: 86px;
        font-weight: 400; 
        text-align: right;
        font-family: 'Space Mono', monospace;
        margin-top: -2px;
        margin-bottom: 5px;
    }
    
    .total-spent-value {
        color: #666666; 
        font-size: 86px; 
        font-weight: 400;
        text-align: right;
        font-family: 'Space Mono', monospace;
        margin-top: -2px;
        margin-bottom: 5px;
    }
    
    .purchasers-count {
        color: #666666; 
        font-size: 86px; 
        font-weight: 400;
        text-align: right;
        font-family: 'Space Mono', monospace;
        margin-top: -2px;
        margin-bottom: 5px;
    }

    /* Added style for CCN count */
    .ccn-companies-count {
        color: #666666; 
        font-size: 86px; 
        font-weight: 400;
        text-align: right;
        font-family: 'Space Mono', monospace;
        margin-top: -2px;
        margin-bottom: 5px;
    }

    .data-source {
        color: #666666;
        font-size: 14px;
        margin-bottom: 10px;
        text-align: left;
    }
    
    .chart-subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 300;
        color: #666666;
        font-size: 36px;
        margin-bottom: 10px; /* Restored margin */
        text-align: left;
        clear: both; /* Ensure it appears below both title and stats */
    }
    
    .highlight-text {
        border-bottom: 3px solid #20DD7B;
        padding-bottom: 2px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .dashboard-container {
            padding: 0 15px;
        }
        
        .chart-container {
            height: 350px;
        }
        
        .delivery-percentage,
        .total-spent-value,
        .purchasers-count,
        .ccn-companies-count { /* Added CCN count */
            font-size: 42px; /* Example adjustment */
        }
        
        .stat-label {
            font-size: 11px;
        }
        
         .chart-title {
            font-size: 12px;
        }
        
        .chart-subtitle {
            font-size: 17px;
        }
        
}

/* Move it here, outside the media query */
.toggle-wrapper {
    display: flex;
    justify-content: flex-end;
    max-width: 1600px;
    margin: 0 auto 10px;
}
</style>
  


<script>
    const API_BASE_URL = 'https://qkx92bmivj.execute-api.ca-central-1.amazonaws.com/test';
    let chartInstance;
    let purchasersChartInstance;
    let ccnChartInstance; // Added variable for CCN chart
    let dealsTickerInstance; // Variable for the deals ticker
    const UPDATE_CHECK_INTERVAL = 4 * 60 * 60 * 1000; // 4 hours like climate dashboard
    let lastUpdateCheck = Date.now();


    document.addEventListener("DOMContentLoaded", () => {
        // Language detection
        var wLocation = window.location.href;
        console.log("This is the url", wLocation)
        if (wLocation.includes("https://fr.deepskyclimate.com") == true) {
            console.log("This is french")
            document.getElementById("french-lang").style.display = "none";
        } else if (wLocation.includes("https://www.deepskyclimate.com") == true) {
            console.log("This is english")
            document.getElementById("english-lang").style.display = "none";
        }

        document.getElementById("english-lang").addEventListener("click", (e)=> {
            e.preventDefault();
        let originalUrl = window.location.href;
        let newUrl = originalUrl.replace("fr.deepskyclimate.com", "www.deepskyclimate.com")
            window.location.href = newUrl;
        })

        document.getElementById("french-lang").addEventListener("click", (e)=> {
            e.preventDefault();
                let originalUrll = window.location.href;
        let newUrll = originalUrll.replace("www.deepskyclimate.com", "fr.deepskyclimate.com")
            window.location.href = newUrll;
        });

        // Device detection
        function waitForElement(selector, callback) {
            const element = document.querySelector(selector);
            if (element) {
                callback(element);
            } else {
                setTimeout(() => waitForElement(selector, callback), 100);
            }
        }
        
        function checkDevice() {
            function isMobile() {
                return document.documentElement.clientWidth < 768;
            }
            if (isMobile()) {
                console.log("Mobile version active");
                waitForElement(".desktop-charts", (desktopElement) => desktopElement.remove());
            } else {
                console.log("Desktop version active");
                waitForElement(".mobile-charts", (mobileElement) => mobileElement.remove());
            }
        }
  
        // Run on load
        checkDevice();
    });

    // Initialize charts and ticker after templates are loaded
    async function initializeCharts() {
        try {
            console.log('Initializing charts and ticker...');

            // --- Initialize Deals Ticker ---
            // Ensure DealsTicker class is available (loaded via chart_templates.js)
            if (typeof DealsTicker !== 'undefined') {
                dealsTickerInstance = new DealsTicker(
                    'dealsTickerContainer', 
                    API_BASE_URL,
                    {
                        dealsKeyPattern: 'processed/latest_deals_*' // Match the pattern used in CDRChart
                    }
                );
                
                // Start the ticker
                dealsTickerInstance.start();
                console.log('Deals Ticker initialized and started.');
            } else {
                console.error('DealsTicker class not found. Ensure chart_templates.js loaded correctly.');
            }
            // --- End Deals Ticker Init ---

            // Create a shared state manager
            const stateManager = new CDRStateManager();

            // Add global state change listener to update all toggles
            stateManager.subscribe((dataType) => {
                // Just call our global sync function - it handles everything
                syncAllToggles();
            });

            // Initialize the main CDR chart
            chartInstance = new CDRChart({
                csvPath: `${API_BASE_URL}/presigned-url?key=processed/orders_for_viz*`,
                toggleContainerId: 'orders-toggle-container', // Updated to specific toggle
                chartContainerId: 'chart-container',
                canvasId: 'cdr-chart',
                statusId: 'status-display',
                enableToggle: true,
                theme: 'light',
                apiBaseUrl: API_BASE_URL,
                chartType: 'orders',
                stateManager: stateManager,
                isStatsProvider: true
            });

            await chartInstance.initialize(); // Initialize orders chart
            console.log('Main CDR chart initialized');

            // Initialize the purchasers chart
            purchasersChartInstance = new CDRChart({
                csvPath: `${API_BASE_URL}/presigned-url?key=processed/purchasers_for_viz*`,
                toggleContainerId: 'purchasers-toggle-container', // Updated to specific toggle
                chartContainerId: 'purchasers-chart-container',
                canvasId: 'purchasers-chart',
                statusId: 'status-display',
                title: 'Unique CDR Purchasers',
                enableToggle: true, // Changed to true to create its own toggle
                theme: 'light',
                apiBaseUrl: API_BASE_URL,
                chartType: 'purchasers',
                stateManager: stateManager,
                isStatsProvider: false
            });

            await purchasersChartInstance.initialize(); // Initialize purchasers chart
            console.log('Purchasers chart initialized');

            // Initialize the CCN chart
            ccnChartInstance = new CDRChart({
                csvPath: `${API_BASE_URL}/presigned-url?key=processed/ccn_data*`,
                toggleContainerId: 'companies-toggle-container', // Updated to specific toggle
                chartContainerId: 'ccn-chart-container',
                canvasId: 'ccn-chart',
                statusId: 'status-display',
                title: 'CDR Companies Founded',
                chartType: 'ccn',
                enableToggle: true, // Changed to true to create its own toggle
                theme: 'light',
                apiBaseUrl: API_BASE_URL,
                stateManager: stateManager,
                isStatsProvider: false
            });

            await ccnChartInstance.initialize(); // Initialize CCN chart
            console.log('CCN chart initialized');

        } catch (error) {
            console.error('Error in initializeCharts:', error);
            // Display error in a designated status area if available
            const statusElement = document.getElementById('status-display');
             if (statusElement) {
                 statusElement.innerHTML =
                     `<div class="error-message">Failed to initialize charts: ${error.message}</div>`;
             }
        }
    }

    // Load chart templates using promise-based approach
    async function loadChartTemplates() {
        try {
            console.log('Fetching chart templates...');
            // Fetch the presigned URL for the chart_templates.js script
            const response = await fetch(`${API_BASE_URL}/presigned-url?key=viz/chart_templates.js`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (!data.url) {
                 throw new Error('Presigned URL for chart templates not found in response.');
            }

            // Create and load script with promise handling
            const script = document.createElement('script');
            script.src = data.url;
            script.crossOrigin = "anonymous"; // Important for loading from S3/CDN

            // Create a promise to handle script loading
            const loadPromise = new Promise((resolve, reject) => {
                script.onload = () => {
                    console.log('Chart templates loaded successfully');
                    // Check if required classes are now available
                    if (typeof CDRChart === 'undefined' || typeof CDRStateManager === 'undefined' || typeof DealsTicker === 'undefined') { // Added DealsTicker check
                         console.error('Required classes (CDRChart, CDRStateManager, DealsTicker) not defined after script load.');
                         reject(new Error('Required classes not defined after script load.'));
                    } else {
                         resolve();
                    }
                };
                script.onerror = (error) => {
                    console.error('Error loading chart templates script:', error);
                    reject(error);
                };
            });

            // Add script to document head
            document.head.appendChild(script);

            // Wait for script to load
            await loadPromise;

            // Initialize charts and ticker after successful load
            await initializeCharts(); // CHANGED back to initializeCharts()

        } catch (error) {
            console.error('Error in loadChartTemplates:', error);
            // Display error in a designated status area if available
            const statusElement = document.getElementById('status-display');
             if (statusElement) {
                 statusElement.innerHTML =
                     `<div class="error-message">Failed to load chart templates: ${error.message}</div>`;
             }
        }
    }

    // Start loading process when DOM is ready
    document.addEventListener('DOMContentLoaded', loadChartTemplates);

    // Check for updates when page becomes visible
    document.addEventListener('visibilitychange', async () => {
        if (document.visibilityState === 'visible') {
            const now = Date.now();
            if (now - lastUpdateCheck > UPDATE_CHECK_INTERVAL) {
                console.log('Page visible, checking for updates...');
                try {
                    // Update all chart instances if they exist
                    if (chartInstance) {
                        await chartInstance.updateChartData('cdr-chart');
                    }
                    if (purchasersChartInstance) {
                        await purchasersChartInstance.updatePurchasersChartData('purchasers-chart');
                    }
                    if (ccnChartInstance) { // Added check and update for CCN chart
                        await ccnChartInstance.updateCCNChartData('ccn-chart');
                    }
                    lastUpdateCheck = now;
                } catch (error) {
                    console.error('Error checking for updates:', error);
                }
            }
        }
    });


    // Global function to sync all toggle states
    function syncAllToggles() {
      // Get the current data type from any of the toggles (assuming at least one exists)
      const checkedToggle = document.querySelector('.ds-toggle-switch input[type="radio"]:checked');
      if (!checkedToggle) return; // No toggles yet
      
      const dataType = checkedToggle.value; // 'all' or 'dac'
      
      // Apply this state to ALL toggle switches
      document.querySelectorAll('.ds-toggle-switch input[type="radio"]').forEach(radio => {
        // Set checked state based on value
        radio.checked = (radio.value === dataType);
        
        // Apply styling to the label
        const label = radio.nextElementSibling;
        if (label) {
          if (radio.checked) {
            label.style.backgroundColor = '#20DD7B';
            label.style.color = 'black';
          } else {
            label.style.backgroundColor = '';
            label.style.color = 'black';
          }
        }
      });
    }

    // Setup a mutation observer to watch for toggles being added
    const toggleObserver = new MutationObserver(mutations => {
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length) {
          // Check if any toggle containers were added
          for (const node of mutation.addedNodes) {
            if (node.nodeType === 1 && (
                node.classList.contains('ds-toggle-container') || 
                node.querySelector('.ds-toggle-container')
            )) {
              // Wait a tiny bit for the DOM to fully process
              setTimeout(syncAllToggles, 50);
              break;
            }
          }
        }
      }
    });

    // Start observing the entire document for toggle additions
    toggleObserver.observe(document.body, { 
      childList: true, 
      subtree: true 
    });

    // Call when document is ready
    document.addEventListener('DOMContentLoaded', () => {
      // Wait a bit for all charts to initialize
      setTimeout(syncAllToggles, 500);
    });

    // Also call it periodically to ensure consistency
    setInterval(syncAllToggles, 2000); // Every 2 seconds
</script>