// Constants for styling
const COLORS = {
    BLUE: '#3B82F6',
    GREEN: '#20DD7B',
    GRAY: '#828282' // Lighter gray for light theme
};

const FONTS = {
    SPACE_MONO: "'Space Mono', monospace",
    IBM: "'IBM Plex Sans', sans-serif",
    SIZE: 12,
    COLOR: '#918F90'
};

class Logger {
    constructor(isDebug = false) { // Default to false in production
        this.isDebug = isDebug;
    }

    log(...args) {
        if (this.isDebug) console.log(...args);
    }

    error(...args) {
        console.error(...args);
    }
    
    debug(...args) {
        if (this.isDebug) console.debug(...args);
    }
    
    warn(...args) {
        if (this.isDebug) console.warn(...args);
    }
}

// Use environment flag for debugging
const logger = new Logger(false); // Set to false in production

function isLanguageFrench() {
    const isFrench = window.location.href.includes('://fr.deepskyclimate.com');
    return isFrench;
}


/**
 * Finds the most recent date-suffixed file matching a base filename pattern
 * @param {string} baseFilePath - Base path without extension (e.g. '../data/processed/orders_for_viz')
 * @returns {Promise<{path: string, date: string|null}>} - Full path to the latest file and extracted date
 */
async function findLatestDateSuffixedFile(baseFilePath) {
    try {
        // If baseFilePath is a presigned URL API endpoint, this function isn't needed
        // The API will handle finding the latest file on the server side
        if (baseFilePath.includes('/presigned-url?key=')) {
            logger.log('Using API to find latest file, no client-side handling needed');
            return {
                path: baseFilePath,
                date: null // We'll get the date later from the CSV content
            };
        }

        // Original implementation for local development
        const dirPath = baseFilePath.substring(0, baseFilePath.lastIndexOf('/'));
        const baseName = baseFilePath.substring(baseFilePath.lastIndexOf('/') + 1);

        logger.log(`Looking for date-suffixed files matching: ${baseName}_ in ${dirPath}`);

        // Try to fetch the directory listing
        const dirResponse = await fetch(dirPath + '/');
        if (!dirResponse.ok) {
            throw new Error(`Failed to access directory: ${dirResponse.statusText}`);
        }

        const dirText = await dirResponse.text();

        // Pattern: baseName_YYYYMMDD.csv
        const regex = new RegExp(`${baseName}_(\\d{8})\\.csv`, 'g');
        const matches = Array.from(dirText.matchAll(regex));

        if (matches && matches.length > 0) {
            // Sort matches by date (descending)
            matches.sort((a, b) => b[1].localeCompare(a[1]));

            const latestFile = matches[0][0];
            const dateStr = matches[0][1];

            // Format date as YYYY-MM-DD
            const formattedDate = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;

            logger.log(`Found latest file: ${latestFile}, date: ${formattedDate}`);
            return {
                path: `${dirPath}/${latestFile}`,
                date: formattedDate
            };
        }

        // Fallback to original filename with .csv
        logger.log(`No date-suffixed files found, trying ${baseFilePath}.csv`);
        return {
            path: `${baseFilePath}.csv`,
            date: null
        };
    } catch (error) {
        logger.error('Error finding latest file:', error);
        return {
            path: `${baseFilePath}.csv`,
            date: null
        };
    }
}

// Keep the global state manager as is
class CDRStateManager {
  constructor() {
    this.dataType = 'all';
    this.listeners = [];
  }

  setDataType(type) {
    if (type !== 'all' && type !== 'dac') {
      console.error(`Invalid data type: ${type}. Must be 'all' or 'dac'`);
      return;
    }
    
    // Add log to confirm state change
    console.log(`CDRStateManager: Setting data type to ${type}`);
    
    this.dataType = type;
    this.notifyListeners();
  }
  
  getDataType() {
    return this.dataType;
  }
  
  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }
  
  notifyListeners() {
    console.log(`CDRStateManager: Notifying ${this.listeners.length} listeners of type: ${this.dataType}`);
    for (const listener of this.listeners) {
      listener(this.dataType);
    }
  }
}

const cdrStateManager = new CDRStateManager();


// Styles for the Deals Ticker
const dealsTickerStyles = `
    .deals-ticker-container {
        width: 100%;
        max-width: 1600px;
        margin: 20px auto;
        overflow: hidden !important; /* Force overflow hidden */
        height: 40px !important; /* Force explicit height */
        background-color: #FFFFFF; /* Changed from #f0f0f0 to white */
        display: block !important; /* Force display block */
        position: relative;
        clear: both;
    }
    
    .deals-ticker-content {
        white-space: nowrap !important;
        overflow: visible !important;
        display: inline-block !important;
        padding: 0; /* Removed top/bottom padding */
        animation: dealsTickerAnim 120s linear infinite !important;
        line-height: 40px !important; /* Match container height for vertical centering */
        height: 100% !important; /* Take full height of container */
    }
    
    .deal-item {
        display: inline-block !important;
        margin-right: 50px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
        color: #333;
        vertical-align: middle !important; /* Ensure text is vertically centered */
    }
    
    /* Ensure animation is defined */
    @keyframes dealsTickerAnim {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
`;

// Update the injection function to ensure styles are applied:
function injectDealsTickerStyles() {
    const styleId = 'cdr-deals-ticker-styles';
    if (!document.getElementById(styleId)) {
        const styleSheet = document.createElement("style");
        styleSheet.id = styleId;
        styleSheet.textContent = dealsTickerStyles;
        document.head.appendChild(styleSheet);
        logger.log('Injected Deals Ticker styles.');
    }
}

// CDR Chart class 
class CDRChart {
    constructor(options = {}) {
        // Only create properties that are actually used
        this.csvPath = options.csvPath || '../data/processed/orders_for_viz';
        this.charts = new Map();
        this.dataType = options.dataType || 'all';
        this.chartType = options.chartType || 'orders';
        this.stateManager = options.stateManager || cdrStateManager;
        this.isStatsProvider = !!options.isStatsProvider;
        
        // Group related properties into objects
        this.config = {
            toggleContainerId: options.toggleContainerId || 'toggle-container',
            chartContainerId: options.chartContainerId || 'chart-container',
            canvasId: options.canvasId || 'cdr-chart',
            statusId: options.statusId || 'status-display',
            enableToggle: options.hasOwnProperty('enableToggle') ? options.enableToggle : true,
            title: options.title || 'State of the Carbon Removal Market'
        };

        this.stats = {
            deliveryPercentages: { all: null, dac: null },
            totalSpent: { all: null, dac: null }
        };
        
        // Other essential properties only
        this.isReady = false;
        this.chartData = null; // Store the full dataset
        this.purchasersData = null; // Store the full dataset for purchasers
        
        // Add lastUpdated property
        this.lastUpdated = null;
        
        // CSS to be injected for the toggle - adjust for light theme
        this.toggleStyles = `
            .ds-toggle-container {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 5px;
            }
            .ds-toggle-switch {
                display: inline-flex;
                background: #E9E9E9;
                border-radius: 40px;
                padding: 5px;
                position: relative;
                height: 60px !important;
                align-items: center;
            }
            .ds-toggle-switch input[type="radio"] {
                display: none;
            }
            .ds-toggle-switch label {
                cursor: pointer;
                padding: 8px 20px !important;
                border-radius: 40px;
                color: black;
                font-family: 'Space Mono', monospace !important;
                font-weight: 300 !important;
                font-size: 16px !important;
                height: 47px !important;
                line-height: 20px !important;
                display: flex;
                align-items: center;
                margin: 0 2px;
            }
            .ds-toggle-switch input[type="radio"]:checked + label {
                background-color: ${COLORS.GREEN};
                color: black;
            }
        `;

        // Add delivery percentage tracking
        this.deliveryPercentages = {
            all: null,
            dac: null
        };
        
        // Add total spent tracking
        this.totalSpent = {
            all: null,
            dac: null
        };

        this.apiBaseUrl = options.apiBaseUrl || '';

        // Add a stateManager property
        this.stateManager = options.stateManager || cdrStateManager;
        
        // Track whether this chart should update the stats
        this.isStatsProvider = options.isStatsProvider || false;
        
        // Get initial data type
        this.dataType = this.stateManager.getDataType();
        
        this.chartType = options.chartType || 'orders'; // Store chart type, default to 'orders'
    }

    // Consolidate all styles into one injection method
    injectStyles() {
        // Check if styles already injected to avoid duplicates
        const styleIds = ['cdr-chart-toggle-styles', 'cdr-deals-ticker-styles'];
        
        for (const id of styleIds) {
            if (document.getElementById(id)) continue;
            
            const style = document.createElement('style');
            style.id = id;
            
            if (id === 'cdr-chart-toggle-styles') {
                style.textContent = this.toggleStyles;
            } else if (id === 'cdr-deals-ticker-styles') {
                style.textContent = dealsTickerStyles;
            }
            
            document.head.appendChild(style);
        }
        
        logger.log('All styles injected');
    }

    async #fetchData(filePath, processFunction) {
        let fetchedPath = filePath; // Keep track of the path used for fetching
        let lastUpdatedDate = null; // Store the date if found locally

        try {
            let csvText;
            // Check if it's a presigned URL API endpoint
            if (filePath.includes('/presigned-url?key=')) {
                logger.log(`Fetching data through presigned URL API: ${filePath}`);
                const response = await fetch(filePath);
                if (!response.ok) throw new Error(`Failed to get presigned URL: ${response.statusText}`);
                
                const data = await response.json();
                if (!data.url) throw new Error('No presigned URL returned from API');
                
                fetchedPath = data.url; // Update path for logging
                logger.log(`Loading data from presigned URL: ${fetchedPath.substring(0, 100)}...`);
                
                // Extract date from the key pattern
                const keyPattern = new URL(filePath).searchParams.get('key');
                console.log(`DEBUG: Extracted key pattern from URL: ${keyPattern}`);
                
                // Try to extract date from the key pattern
                if (keyPattern) {
                    const matches = keyPattern.match(/(\d{8})/);
                    if (matches && matches[1]) {
                        // Format date as YYYY-MM-DD
                        const dateStr = matches[1];
                        lastUpdatedDate = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
                        console.log(`DEBUG: Extracted date from S3 key pattern: ${lastUpdatedDate}`);
                    } else {
                        // If no date in key pattern, try to find it in the response headers or S3 URL
                        const today = new Date();
                        lastUpdatedDate = today.toISOString().split('T')[0]; // Use today as fallback
                        console.log(`DEBUG: Using today's date as fallback: ${lastUpdatedDate}`);
                    }
                }
                
                const dataResponse = await fetch(fetchedPath);
                if (!dataResponse.ok) throw new Error(`Failed to fetch data: ${dataResponse.statusText}`);
                csvText = await dataResponse.text(); // FIXED: Use dataResponse, not response
                
            } else {
                // Handle local development file fetching
                let actualPath = filePath;
                if (!filePath.endsWith('.csv')) {
                    // Try to find latest date-suffixed file using the base path
                    const result = await findLatestDateSuffixedFile(filePath); // Pass base path
                    actualPath = result.path;
                    lastUpdatedDate = result.date; // Store the date found
                } else {
                    // If a specific .csv is provided, use it directly
                    actualPath = filePath;
                }
                
                fetchedPath = actualPath; // Update path for logging
                logger.log(`Loading data from local path: ${fetchedPath}`);
                const response = await fetch(fetchedPath);
                if (!response.ok) throw new Error(`Failed to fetch data: ${response.statusText}`);
                
                csvText = await response.text();
            }

            // Process the fetched text using the provided function
            const processedData = processFunction.call(this, csvText); // Call the process function with the correct context
            return { data: processedData, lastUpdated: lastUpdatedDate };

        } catch (error) {
            logger.error(`Error fetching data from ${fetchedPath}:`, error);
            return { data: null, lastUpdated: lastUpdatedDate, error: error }; 
        }
    }

    getScaleConfig(labels, chartType = 'orders') {
        const isTimeSeries = chartType === 'orders' || chartType === 'purchasers' || chartType === 'carbonPricing';
        
        const scaleConfig = {
            x: {
                grid: {
                    display: false
                },
                border: {
                    display: true
                },
                ticks: {
                    color: COLORS.GRAY,
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: FONTS.SIZE,
                        // weight: 'bold'
                    },
                    maxRotation: 0,
                    minRotation: 0,
                }
            },
            y: {
                position: 'right',
                grid: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.2)' // Use light grid line color
                },
                border: {
                    display: false
                },
                ticks: {
                    color: COLORS.GRAY,
                    font: {
                        family: FONTS.SPACE_MONO, // Changed from FONTS.FAMILY
                        size: FONTS.SIZE,
                        // weight: 'bold'
                    },
                    padding: 4 // Add padding between labels and the chart area
                }
            }
        };

        // Configure X-axis specifically for time-series charts (orders, purchasers)
        if (isTimeSeries) {
            scaleConfig.x.type = 'time';
            scaleConfig.x.time = {
                unit: 'year', // Display major ticks at the start of each year
                tooltipFormat: 'MMM d, yyyy', // Format for tooltips
                displayFormats: {
                    year: 'yyyy' // Display only the year on the axis ticks
                }
            };
            // Adjust ticks for time series
            scaleConfig.x.ticks.source = 'auto'; // Let Chart.js determine ticks based on unit
            scaleConfig.x.ticks.autoSkip = true; // Allow skipping if years get too dense
            scaleConfig.x.ticks.maxTicksLimit = undefined; // Remove limit for time scale
             scaleConfig.x.ticks.major = {
                 enabled: true // Ensure major ticks are enabled for the 'year' unit
             };
             // Remove callback for time series
             delete scaleConfig.x.ticks.callback; 

        }
        
        // Specific Y-axis formatting based on chart type
        if (chartType === 'orders') {
            scaleConfig.y.ticks.callback = value => `${Math.round(value / 1_000_000)}M`;
        } else if (chartType === 'purchasers' || chartType === 'carbonPricing') {
             scaleConfig.y.ticks.callback = value => `${Math.round(value)}`;
        }

        return scaleConfig;
    }

    addLoadingIndicator(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'chart-loading';
            loadingIndicator.textContent = 'Pulling latest data...';
            loadingIndicator.style.cssText = 
                'position: absolute; top: 50%; left: 50%; ' +
                'transform: translate(-50%, -50%); color: #929190; ' + 
                'font-family: "Space Mono", monospace;';
            container.style.position = 'relative';
            container.appendChild(loadingIndicator);
            return loadingIndicator;
        }
        return null;
    }
    
    // Create the toggle UI with unique IDs based on chart type
    createToggleUI() {
        if (!this.config.enableToggle) return;
        
        const toggleContainer = document.getElementById(this.config.toggleContainerId);
        if (!toggleContainer) {
            logger.error(`Toggle container not found: ${this.config.toggleContainerId}`);
            return;
        }

        // Clear any existing content to avoid duplicate toggles
        toggleContainer.innerHTML = '';

        // Generate unique IDs based on chart type
        const chartTypeSuffix = this.chartType || 'chart';
        const allCdrId = `all-cdr-${chartTypeSuffix}`;
        const dacOnlyId = `dac-only-${chartTypeSuffix}`;
        
        // Determine which language to use
        const isFrench = isLanguageFrench();
        
        // Set labels based on language
        const allCdrLabel = isFrench ? 'TOUT EDC' : 'ALL CDR';
        const dacOnlyLabel = isFrench ? 'JUSTE CDA' : 'DAC ONLY';
        
        // Create the toggle switch UI
        const fragment = document.createDocumentFragment();
    
        // Create elements and append to fragment
        const toggleWrapper = document.createElement('div');
        toggleWrapper.className = 'ds-toggle-container';
        
        const toggleSwitch = document.createElement('div');
        toggleSwitch.className = 'ds-toggle-switch';
        toggleSwitch.innerHTML = `
            <input type="radio" id="${allCdrId}" name="cdr-type-${chartTypeSuffix}" value="all" ${this.dataType === 'all' ? 'checked' : ''}>
            <label for="${allCdrId}">${allCdrLabel}</label>
            <input type="radio" id="${dacOnlyId}" name="cdr-type-${chartTypeSuffix}" value="dac" ${this.dataType === 'dac' ? 'checked' : ''}>
            <label for="${dacOnlyId}">${dacOnlyLabel}</label>
        `;
        
        toggleWrapper.appendChild(toggleSwitch);
        fragment.appendChild(toggleWrapper);
        
        // Append fragment to DOM (single reflow)
        toggleContainer.appendChild(fragment);
        
        // Store the radio button elements for updating when state changes
        this.allCdrRadio = document.getElementById(allCdrId);
        this.dacOnlyRadio = document.getElementById(dacOnlyId);
        
        // Ensure toggle styles are applied initially
        this.updateToggleStyles();
        
        // Add event listeners directly to radio buttons, not the container
        this.allCdrRadio.addEventListener('change', () => {
            console.log(`[${this.chartType}] Toggle changed to: all`);
            this.stateManager.setDataType('all');
        });
        
        this.dacOnlyRadio.addEventListener('change', () => {
            console.log(`[${this.chartType}] Toggle changed to: dac`);
            this.stateManager.setDataType('dac');
        });
    }

    // Add this helper method:
    updateToggleStyles() {
        if (!this.allCdrRadio || !this.dacOnlyRadio) return;
        
        // Get the labels associated with the radio buttons
        const allCdrLabel = this.allCdrRadio.nextElementSibling;
        const dacOnlyLabel = this.dacOnlyRadio.nextElementSibling;
        
        if (!allCdrLabel || !dacOnlyLabel) return;
        
        // Update checked state based on current data type
        this.allCdrRadio.checked = this.dataType === 'all';
        this.dacOnlyRadio.checked = this.dataType === 'dac';
        
        // Update styles based on current data type
        allCdrLabel.style.backgroundColor = this.dataType === 'all' ? COLORS.GREEN : '';
        dacOnlyLabel.style.backgroundColor = this.dataType === 'dac' ? COLORS.GREEN : '';
    }

    processOrdersData(csvText) {
        // Parse CSV
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        // Find column indices
        const dateIndex = headers.indexOf('date');
        const purchasedIndex = headers.indexOf('tons_purchased_cum');
        const deliveredIndex = headers.indexOf('tons_delivered_cum');
        const dacPurchasedIndex = headers.indexOf('dac_tons_purchased_cum');
        const dacDeliveredIndex = headers.indexOf('dac_tons_delivered_cum');
        const spentIndex = headers.indexOf('price_usd_cum');
        const dacSpentIndex = headers.indexOf('dac_price_usd_cum');
        
        if (dateIndex === -1 || purchasedIndex === -1 || deliveredIndex === -1 || 
            dacPurchasedIndex === -1 || dacDeliveredIndex === -1) {
            logger.error(`Required columns not found in CSV. Headers: ${headers.join(', ')}`);
            return { 
                labels: [], 
                purchased: [], 
                delivered: [],
                dacPurchased: [], 
                dacDelivered: [] 
            };
        }

        // Process rows (data is already cumulative in the CSV)
        const data = lines.slice(1)
            .filter(row => row.trim())
            .map(row => {
                const columns = row.split(',');
                return {
                    date: columns[dateIndex], // Check this format
                    purchased: parseFloat(columns[purchasedIndex]) || 0,
                    delivered: parseFloat(columns[deliveredIndex]) || 0,
                    dacPurchased: parseFloat(columns[dacPurchasedIndex]) || 0,
                    dacDelivered: parseFloat(columns[dacDeliveredIndex]) || 0,
                    usdSpent: parseFloat(columns[spentIndex]) || 0,
                    dacUsdSpent: parseFloat(columns[dacSpentIndex]) || 0
                };
            })
            .sort((a, b) => new Date(a.date) - new Date(b.date));
        
        if (data.length > 0) {
            logger.log('[processOrdersData] Sample processed date:', data[0].date); // Log first date format
        }

        // Extract arrays for charting
        const labels = data.map(row => row.date);
        logger.log('[processOrdersData] Final labels sample:', labels.slice(0, 5), '...'); // Log final labels

        const purchased = data.map(row => row.purchased);
        const delivered = data.map(row => row.delivered);
        const dacPurchased = data.map(row => row.dacPurchased);
        const dacDelivered = data.map(row => row.dacDelivered);

        // Calculate delivery percentages from the latest data point
        if (data.length > 0) {
            const latest = data[data.length - 1];
            this.deliveryPercentages.all = latest.purchased > 0 ? 
                (latest.delivered / latest.purchased * 100) : 0;
            this.deliveryPercentages.dac = latest.dacPurchased > 0 ? 
                (latest.dacDelivered / latest.dacPurchased * 100) : 0;
            // MDK I have commented out the totalSpent.all line directly below in order to use the scraped dollars spent value rather than the one from the actual data. 
            // this.totalSpent.all = latest.usdSpent;
            this.totalSpent.dac = latest.dacUsdSpent;
        }

        return { 
            labels, 
            purchased, 
            delivered, 
            dacPurchased, 
            dacDelivered 
        };
    }

    processPurchasersData(csvText) {
        // Parse CSV
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        // Find column indices
        const dateIndex = headers.indexOf('date');
        const purchasersIndex = headers.indexOf('purchasers_count_cum');
        const dacPurchasersIndex = headers.indexOf('dac_purchasers_count_cum');
        
        if (dateIndex === -1 || purchasersIndex === -1 || dacPurchasersIndex === -1) {
            logger.error(`Required columns not found in CSV. Headers: ${headers.join(', ')}`);
            return { 
                labels: [], 
                purchasers: [], 
                dacPurchasers: []
            };
        }

        // Process rows
        const data = lines.slice(1)
            .filter(row => row.trim())
            .map(row => {
                const columns = row.split(',');
                return {
                    date: columns[dateIndex], // Check this format
                    purchasers: parseFloat(columns[purchasersIndex]) || 0,
                    dacPurchasers: parseFloat(columns[dacPurchasersIndex]) || 0
                };
            })
            .sort((a, b) => new Date(a.date) - new Date(b.date));
        
        if (data.length > 0) {
            logger.log('[processPurchasersData] Sample processed date:', data[0].date); // Log first date format
        }

        // Extract arrays for charting
        const labels = data.map(row => row.date);
        logger.log('[processPurchasersData] Final labels sample:', labels.slice(0, 5), '...'); // Log final labels

        const purchasers = data.map(row => row.purchasers);
        const dacPurchasers = data.map(row => row.dacPurchasers);

        return { labels, purchasers, dacPurchasers };
    }

    processCarbonPricingData(csvText) {
        // Parse CSV
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        // Find column indices
        const yearIndex = headers.indexOf('year');
        const countriesCountIndex = headers.indexOf('countries_count');
        const emissionsPctIndex = headers.indexOf('global_emissions_pct');
        
        if (yearIndex === -1 || countriesCountIndex === -1 || emissionsPctIndex === -1) {
            logger.error(`Required columns not found in CSV. Headers: ${headers.join(', ')}`);
            return { 
                labels: [], 
                carbonPricingCountries: [],
                globalEmissionsPct: []
            };
        }
    
        // Process rows
        const data = lines.slice(1)
            .filter(row => row.trim())
            .map(row => {
                const columns = row.split(',');
                return {
                    year: columns[yearIndex], // Year as string (e.g. "2010")
                    countriesCount: parseFloat(columns[countriesCountIndex]) || 0,
                    emissionsPct: parseFloat(columns[emissionsPctIndex]) || 0
                };
            })
            .sort((a, b) => parseInt(a.year) - parseInt(b.year)); // Sort by year
        
        if (data.length > 0) {
            logger.log('[processCarbonPricingData] Sample processed year:', data[0].year);
        }
    
        // Extract arrays for charting - convert years to Date objects for consistent handling
        const labels = data.map(row => `${row.year}-01-01`); // Format as ISO dates (Jan 1 of each year)
        const carbonPricingCountries = data.map(row => row.countriesCount);
        const globalEmissionsPct = data.map(row => row.emissionsPct);
    
        logger.log('[processCarbonPricingData] Final labels sample:', labels.slice(0, 5), '...');
    
        return { 
            labels, 
            carbonPricingCountries,
            globalEmissionsPct
        };
    }

    // Only load and process data when needed
    async initialize() {
        // Inject CSS styles first
        this.injectStyles();
        
        // Add loading indicator
        const loadingIndicator = this.addLoadingIndicator(this.config.chartContainerId);
        
        try {
            // Create toggle UI if enabled
            this.createToggleUI();
            
            // Subscribe to state changes
            this.stateManager.subscribe((dataType) => {
                console.log(`[${this.chartType}] State change detected: ${dataType}`);
                this.handleDataTypeChange(dataType);
            });
            
            // Only load the data and create the chart for the current type
            await this.createChart(this.chartType, this.config.canvasId);
            
            // Common initialization continues...
        } catch (error) {
            // Error handling
        } finally {
            // Cleanup
        }
    }

    async createOrdersDeliveriesChart(canvasId) {
        // Fetch data and store it for later use
        this.chartData = await this.fetchDataByType('orders');
        
        if (!this.chartData || this.chartData.labels.length === 0) { // Added check for this.chartData
            logger.error('No data available to create orders chart');
            return;
        }
        
        logger.log('[createOrdersDeliveriesChart] Labels:', this.chartData.labels.slice(0, 5), '...'); // Log first 5 labels
        logger.log('[createOrdersDeliveriesChart] Data length:', this.chartData.labels.length);

        this.removeLoadingIndicator(this.config.chartContainerId);

        const canvas = document.getElementById(canvasId);

        // Get the appropriate data based on current data type
        const { purchasedData, deliveredData } = this.getCurrentData();

        // Get shared scale configuration
        const scales = this.getScaleConfig(this.chartData.labels, 'orders');

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                layout: {
                    padding: {
                        left: 0, // Ensure the legend starts at the left edge
                        bottom: 5
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    displayColors: true,
                    position: 'nearest', // Add this line
                    caretPadding: 10,    // Add this for better positioning
                    yAlign: 'top',       // Add this to align with top line
                    titleFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    bodyFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    callbacks: {
                        label: context => {
                            const value = context.parsed.y;
                            // Get translated label
                            const translatedLabel = this.getTranslation(context.dataset.label);
                            
                            // Handle different ranges of values
                            if (value < 1_000) {
                                return `${translatedLabel}: ${Math.round(value)} ${this.getTranslation('TONS')}`; 
                            } else if (value < 1_000_000) {
                                return `${translatedLabel}: ${Math.round(value / 1_000)} ${this.getTranslation('K TONS')}`;
                            } else {
                                return `${translatedLabel}: ${Math.round(value / 1_000_000)} ${this.getTranslation('M TONS')}`;
                            }
                        }
                    }
                }
            },
            scales: scales, // Use the generated scales
        };

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: this.chartData.labels, 
                // --- Restore datasets ---
                datasets: [
                    {
                        label: this.getTranslation('TONS PURCHASED'),
                        data: purchasedData,
                        borderColor: COLORS.GRAY,
                        backgroundColor: 'transparent',
                        borderWidth: 4,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: this.getTranslation('TONS DELIVERED'),
                        data: deliveredData,
                        borderColor: COLORS.GREEN,
                        backgroundColor: 'transparent',
                        borderWidth: 4,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    }
                ]
                // --- End restore datasets ---
            },
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        logger.log('Orders/Deliveries chart created successfully');
        if (this.isStatsProvider) {
            await this.fetchDollarsSpent();

            // Update delivery percentage
            const deliveryPerc = this.calculateDeliveryPercentage();
            this.updateStatDisplay('.delivery-percentage', deliveryPerc !== null ? 
                (deliveryPerc < 1 ? '<1%' : `${Math.round(deliveryPerc)}%`) : null);
            
            // Update total spent
            const spentValue = this.calculateTotalSpent();
            this.updateStatDisplay('.total-spent-value', spentValue !== null ? spentValue : null);
            
        }
        // Update last updated date elements
        this.updateLastUpdatedElements(this.lastUpdated);
        
        return chart;
    }


    async createPurchasersChart(canvasId) {
        // Fetch data and store it for later use
        this.purchasersData = await this.fetchDataByType('purchasers');
        
        if (!this.purchasersData || this.purchasersData.labels.length === 0) { // Added check for this.purchasersData
            logger.error('No data available to create purchasers chart');
            return;
        }
        
        logger.log('[createPurchasersChart] Labels:', this.purchasersData.labels.slice(0, 5), '...'); // Log first 5 labels
        logger.log('[createPurchasersChart] Data length:', this.purchasersData.labels.length);

        this.removeLoadingIndicator(this.config.chartContainerId);

        const canvas = document.getElementById(canvasId);

        // Get the appropriate data based on current data type
        const purchasersData = this.dataType === 'dac' ? 
            this.purchasersData.dacPurchasers : 
            this.purchasersData.purchasers;

        // Get shared scale configuration
        const scales = this.getScaleConfig(this.purchasersData.labels, 'purchasers');

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                layout: {
                    padding: {
                        left: 0, // Ensure the legend starts at the left edge
                        bottom: 5
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    displayColors: true,
                    titleFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    bodyFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    callbacks: {
                        label: context => {
                            const value = context.parsed.y;
                            const translatedLabel = this.getTranslation(context.dataset.label);
                            return `${translatedLabel}: ${Math.round(value)}`;
                        }
                    }
                }
            },
            scales: scales
        };

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: this.purchasersData.labels, 
                datasets: [
                    {
                        label: this.getTranslation('CUMULATIVE UNIQUE PURCHASERS'),
                        data: purchasersData,
                        borderColor: COLORS.GRAY,
                        backgroundColor: 'transparent',
                        borderWidth: 4,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        logger.log('Purchasers chart created successfully');
        
        // Update the purchasers count
        const latestPurchasers = this.getLatestValue(this.purchasersData, 'purchasers', 'dacPurchasers');
        this.updateStatDisplay('.purchasers-count', latestPurchasers !== null ? 
            Math.round(latestPurchasers) : null);
        
        return chart;
    }

    async createCarbonPricingCountriesChart(canvasId) {
        // Fetch data and store it
        this.carbonPricingData = await this.fetchDataByType('carbonPricing');
        
        if (!this.carbonPricingData || this.carbonPricingData.labels.length === 0) {
            logger.error('No data available to create carbon pricing countries chart');
            return;
        }
        
        logger.log('[createCarbonPricingCountriesChart] Labels:', this.carbonPricingData.labels.slice(0, 5), '...');
        logger.log('[createCarbonPricingCountriesChart] Data length:', this.carbonPricingData.labels.length);

        this.removeLoadingIndicator(this.config.chartContainerId);

        const canvas = document.getElementById(canvasId);

        // Get the country count and emissions percentage data
        const carbonPricingCountries = this.carbonPricingData.carbonPricingCountries;
        const globalEmissionsPct = this.carbonPricingData.globalEmissionsPct;

        // Configure scales for dual axes
        const scales = {
            x: {
                type: 'time',
                time: {
                    unit: 'year',
                    tooltipFormat: 'yyyy',
                    displayFormats: {
                        year: 'yyyy'
                    }
                },
                grid: {
                    display: false
                },
                border: {
                    display: true
                },
                ticks: {
                    color: COLORS.GRAY,
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: FONTS.SIZE
                    },
                    maxRotation: 0,
                    minRotation: 0,
                }
            },
            y: {
                type: 'linear',
                position: 'left',
                grid: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.2)'
                },
                border: {
                    display: false
                },
                title: {
                    display: true,
                    text: this.getTranslation('NUMBER OF COUNTRIES'),
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: 12
                    },
                    color: COLORS.GRAY
                },
                ticks: {
                    color: COLORS.GRAY,
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: FONTS.SIZE,
                        // weight: 'bold'
                    },
                    padding: 2,
                    callback: value => `${Math.round(value)}`
                }
            },
            y1: {
                type: 'linear',
                position: 'right',
                grid: {
                    display: false
                },
                border: {
                    display: false
                },
                title: {
                    display: true,
                    text: this.getTranslation('% OF GLOBAL EMISSIONS'),
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: 12
                    },
                    color: COLORS.GRAY
                    // padding: {
                    //     top: 10,
                    //     bottom: 10,
                    //     left: 2,
                    //     right: 2
                    // }
                },
                ticks: {
                    color: COLORS.GRAY,
                    font: {
                        family: FONTS.SPACE_MONO,
                        size: FONTS.SIZE,
                        // weight: 'bold'
                    },
                    padding: 2,
                    callback: value => `${value}%`
                }
            }
        };

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                    position: 'top',
                    align: 'start',
                    labels: {
                        font: {
                            family: FONTS.SPACE_MONO,
                            size: 14
                        },
                        color: COLORS.GRAY,
                        boxWidth: 15,
                        padding: 35,
                        usePointStyle: true, // CHANGE 3: Use point style instead of boxes
                        pointStyle: 'line',
                        sort: (a, b) => a.datasetIndex - b.datasetIndex // Ensure legend items follow dataset order
                    },
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    displayColors: true,
                    titleFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    bodyFont: { family: FONTS.SPACE_MONO, size: FONTS.SIZE },
                    callbacks: {
                        label: context => {
                            const value = context.parsed.y;
                            const translatedLabel = this.getTranslation(context.dataset.label);
                            
                            // Format based on dataset
                            if (context.dataset.yAxisID === 'y') {
                                // Countries count
                                return `${translatedLabel}: ${Math.round(value)}`;
                            } else {
                                // Emissions percentage
                                return `${translatedLabel}: ${value.toFixed(1)}%`;
                            }
                        }
                    }
                }
            },
            scales: scales
        };

        const chart = new Chart(canvas, {
            type: 'bar', // Default type for the whole chart
            data: {
                labels: this.carbonPricingData.labels,
                datasets: [
                    {
                        type: 'bar',
                        label: this.getTranslation('COUNTRIES WITH CARBON PRICING'),
                        data: carbonPricingCountries,
                        backgroundColor: COLORS.GRAY,
                        borderColor: COLORS.GRAY,
                        borderWidth: 0,
                        yAxisID: 'y',
                        order: 2 // Higher order means it's drawn first (behind the line)
                    },
                    {
                        type: 'line',
                        label: this.getTranslation('% OF GLOBAL EMISSIONS'),
                        data: globalEmissionsPct,
                        borderColor: COLORS.GREEN,
                        backgroundColor: 'transparent',
                        borderWidth: 4,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        yAxisID: 'y1',
                        order: 1 // Lower order means it's drawn last (on top)
                    }
                ]
            },
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        logger.log('Carbon pricing chart created successfully');
        
        // Update the countries count stat - get the most recent count
        const latestCountriesCount = carbonPricingCountries[carbonPricingCountries.length - 1];
        this.updateStatDisplay('.carbon-pricing-countries-count', latestCountriesCount !== null ? 
            Math.round(latestCountriesCount) : null);
        
        // Add a new stat for global emissions coverage if needed
        const latestEmissionsPct = globalEmissionsPct[globalEmissionsPct.length - 1];
        this.updateStatDisplay('.global-emissions-coverage', latestEmissionsPct !== null ? 
            `${Math.round(latestEmissionsPct)}%` : null);        
        return chart;
    }
    
    // Helper method to get the appropriate data based on data type
    getCurrentData() {
        if (!this.chartData) {
            return { purchasedData: [], deliveredData: [] };
        }
        
        if (this.dataType === 'dac') {
            return {
                purchasedData: this.chartData.dacPurchased,
                deliveredData: this.chartData.dacDelivered
            };
        } else {
            return {
                purchasedData: this.chartData.purchased,
                deliveredData: this.chartData.delivered
            };
        }
    }
    
    // Add method to get delivery percentages
    getDeliveryPercentages() {
        return {
            all: this.deliveryPercentages.all !== null ? 
                Math.round(this.deliveryPercentages.all) : null,
            dac: this.deliveryPercentages.dac !== null ? 
                Math.round(this.deliveryPercentages.dac) : null
        };
    }
    
    // Add method to get total spent
    getTotalSpent() {
        return {
            all: this.totalSpent.all !== null ? 
                this.formatCurrencyValue(this.totalSpent.all) : null,
            dac: this.totalSpent.dac !== null ? 
                this.formatCurrencyValue(this.totalSpent.dac) : null
        };
    }

    // Helper method to format currency values
    formatCurrencyValue(value) {
        if (value >= 1_000_000_000) {
            // Format billions with one decimal place
            return `$${(value / 1_000_000_000).toFixed(1).replace(/\.0$/, '')}B`;
        } else if (value >= 1_000_000) {
            return `$${Math.round(value / 1_000_000)}M`;
        } else if (value >= 1_000) {
            return `$${Math.round(value / 1_000)}K`;
        } else {
            return `$${Math.round(value)}`;
        }
    }   
    
    setupResizeListener() {
        const chartContainer = document.getElementById(this.config.chartContainerId);
        if (!chartContainer) return;
        
        // Create a ResizeObserver to handle container resizing
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                // Update all charts when container size changes
                for (const [canvasId, chart] of this.charts.entries()) {
                    if (chart && typeof chart.resize === 'function') {
                        chart.resize();
                        chart.update('none'); // Update without animation
                    }
                }
            }
        });
        
        // Start observing
        resizeObserver.observe(chartContainer);
        
        // Store reference for cleanup
        this.resizeObserver = resizeObserver;
    }

    // Update this method to support both chart types
    updateSubtitleText() {
        // Update all subtitle elements with the technology class
        document.querySelectorAll('[id$="subtitle-technology"]').forEach(subtitleTech => {
            if (subtitleTech) {
                subtitleTech.textContent = this.dataType === 'dac' ? 'Direct air capture ' : 'Carbon removal   ';
            }
        });
    }

    getTranslation(englishText) {
        const isFrench = isLanguageFrench();
        if (!isFrench) return englishText;
        
        const translations = {
            'TONS PURCHASED': 'TONNES ACHETÉES',
            'TONS DELIVERED': 'TONNES LIVRÉES',
            'CUMULATIVE UNIQUE PURCHASERS': 'ACHETEURS UNIQUES CUMULÉS',
            'COUNTRIES WITH CARBON PRICING': 'PAYS AVEC TARIFICATION DU CARBONE',
            'GLOBAL EMISSIONS COVERAGE': 'COUVERTURE DES ÉMISSIONS MONDIALES',
            'Countries': 'Pays',
            '% of Global Emissions': '% des émissions mondiales',
            'TONS': 'TONNES',
            'K TONS': 'K TONNES',
            'M TONS': 'M TONNES'
        };
        
        return translations[englishText] || englishText;
    }

    // Generic method to update a stat display element
    updateStatDisplay(selector, value, fallbackValue = '--') {
        const element = document.querySelector(selector);
        if (!element) {
            logger.debug(`Stat display element not found: ${selector}, skipping update`);
            return;
        }

        // Check if value is valid (not null, undefined, or NaN)
        const isValidValue = value !== null && value !== undefined && !(typeof value === 'number' && isNaN(value));

        element.textContent = isValidValue ? value : fallbackValue;
    }
    handleDataTypeChange(dataType) {
        console.log(`[${this.chartType}] Handling data type change: ${dataType}`);
    
        if (dataType === this.dataType) {
            console.log(`[${this.chartType}] Data type unchanged, skipping update`);
            return; // Skip if no actual change
        }
        
        // Important: Set isReady to true if it's not already - fixes the early return issue
        this.isReady = true;
        
        this.dataType = dataType; // Update internal state ONCE
        
        // Update the toggle UI to match the state
        this.updateToggleStyles();
        
        // Chart update logic starts here - no early return
        const chart = this.charts.get(this.config.canvasId); 
        if (chart) {
            if (this.chartType === 'orders') { 
                const { purchasedData, deliveredData } = this.getCurrentData(); 
                chart.data.datasets[0].data = purchasedData;
                chart.data.datasets[1].data = deliveredData;
            } else if (this.chartType === 'purchasers') { 
                if (this.purchasersData) { 
                    const purchasersData = dataType === 'dac' ? 
                        this.purchasersData.dacPurchasers : 
                        this.purchasersData.purchasers;
                    chart.data.datasets[0].data = purchasersData;
                }
            }
            
            chart.update(); // Important: Actually update the chart
        }
        
        // Update stats and UI for the specific chart types
        if (this.chartType === 'orders' && this.isStatsProvider) {
            // Update delivery percentage
            const deliveryPerc = this.calculateDeliveryPercentage();
            this.updateStatDisplay('.delivery-percentage', deliveryPerc !== null ? 
                (deliveryPerc < 1 ? '<1%' : `${Math.round(deliveryPerc)}%`) : null);
            
            // Update total spent
            const spentValue = this.calculateTotalSpent();
            this.updateStatDisplay('.total-spent-value', spentValue !== null ? spentValue : null);
        }
        
        if (this.chartType === 'purchasers') {
            // Update purchasers count
            const latestPurchasers = this.getLatestValue(this.purchasersData, 'purchasers', 'dacPurchasers');
            this.updateStatDisplay('.purchasers-count', latestPurchasers !== null ? 
                Math.round(latestPurchasers) : null);
        }
    
        // Always update subtitle text
        this.updateSubtitleText(); 
    
        logger.log(`Changed data type to: ${dataType} for chart type: ${this.chartType}`);
    }

    // Helper to get the latest value from a data array based on dataType
    getLatestValue(dataObject, allKey, dacKey) {
        if (!dataObject) return null;

        const dataArray = this.dataType === 'dac' ? dataObject[dacKey] : dataObject[allKey];

        if (!dataArray || dataArray.length === 0) return null;

        return dataArray[dataArray.length - 1];
    }

    // Helper to calculate the current delivery percentage
    calculateDeliveryPercentage() {
        if (!this.chartData || this.chartData.labels.length === 0) return null;

        const latestIndex = this.chartData.labels.length - 1;
        let delivered, purchased;

        if (this.dataType === 'dac') {
            delivered = this.chartData.dacDelivered[latestIndex];
            purchased = this.chartData.dacPurchased[latestIndex];
        } else {
            delivered = this.chartData.delivered[latestIndex];
            purchased = this.chartData.purchased[latestIndex];
        }

        if (purchased > 0) {
            return (delivered / purchased * 100);
        }
        return 0; // Return 0 if nothing purchased
    }

    // Helper to calculate the current total spent value
    calculateTotalSpent() {
        if (!this.totalSpent) return null;

        const value = this.dataType === 'dac' ? this.totalSpent.dac : this.totalSpent.all;
        if (value === null || value === undefined) return null;
        
        // Format the value properly
        return this.formatCurrencyValue(value);
    }

    async fetchDollarsSpent() {
        // This method can't be removed or refactored since it handles a special case
        // Keep the implementation as is
        try {
            // Use findLatestDateSuffixedFile which handles finding files with date suffixes
            const result = await findLatestDateSuffixedFile(this.csvPath.replace('orders_for_viz', 'dollars_spent'));
            logger.log(`Loading dollars spent from: ${result.path}`);
            
            // Check if the response is from the presigned URL API
            let csvText;
            let response = await fetch(result.path);
            
            if (!response.ok) {
                throw new Error(`Failed to load dollars spent data: ${response.statusText}`);
            }
            
            let responseText = await response.text();
            logger.log(`Received response: ${responseText.substring(0, 100)}...`); // Log the start of the response
            
            // Check if the response is JSON (from presigned URL API)
            if (responseText.trim().startsWith('{') && responseText.includes('"url"')) {
                try {
                    // Parse the JSON response
                    const jsonResponse = JSON.parse(responseText);
                    // Fetch the actual CSV from the presigned URL
                    logger.log(`Following presigned URL: ${jsonResponse.url.substring(0, 100)}...`);
                    
                    const csvResponse = await fetch(jsonResponse.url);
                    if (!csvResponse.ok) {
                        throw new Error(`Failed to load data from presigned URL: ${csvResponse.statusText}`);
                    }
                    
                    csvText = await csvResponse.text();
                } catch (e) {
                    logger.error(`Error processing presigned URL: ${e}`);
                    throw e;
                }
            } else {
                // Direct CSV response
                csvText = responseText;
            }
            
            logger.log(`Parsed CSV content: ${csvText}`);
            
            // Parse CSV (it's a simple 2-column file)
            const lines = csvText.split('\n').filter(row => row.trim());
            if (lines.length < 2) {
                throw new Error('CSV data is empty or missing data');
            }
            
            // Get the value from the second line, second column
            const values = lines[1].split(',');
            const value = parseFloat(values[1]);
            if (isNaN(value)) {
                throw new Error('Invalid dollars spent value');
            }
            
            logger.log(`Successfully loaded dollars spent: ${value}`);
            
            // Store the exact value without any formatting, but ONLY for the 'all' key
            // This preserves the DAC value from the orders data
            this.totalSpent.all = value;
            
            return value;
        } catch (error) {
            logger.error('Error fetching dollars spent:', error);
            return null;
        }
    }

    async fetchDataByType(dataType) {
        const pathMap = {
            'orders': this.csvPath,
            'purchasers': this.csvPath.replace('orders_for_viz', 'purchasers_for_viz'),
            'carbonPricing': this.csvPath.replace('orders_for_viz', 'carbon_pricing')
        };
        
        const processorMap = {
            'orders': this.processOrdersData,
            'purchasers': this.processPurchasersData,
            'carbonPricing': this.processCarbonPricingData
        };
        
        const path = pathMap[dataType];
        const processor = processorMap[dataType];
        
        if (!path || !processor) {
            logger.error(`Invalid data type: ${dataType}`);
            return null;
        }
        
        const result = await this.#fetchData(path, processor);
        
        // Update last updated date if one was found
        if (result.lastUpdated) {
            this.updateLastUpdatedElements(result.lastUpdated);
        }
        
        return result.data;
    }

    // Create a single method that handles chart creation based on type
    async createChart(chartType, canvasId) {
        switch (chartType) {
            case 'orders':
                return this.createOrdersDeliveriesChart(canvasId);
            case 'purchasers':
                return this.createPurchasersChart(canvasId);
            case 'carbonPricing':
                return this.createCarbonPricingCountriesChart(canvasId);
            default:
                logger.error(`Unknown chart type: ${chartType}`);
                return null;
        }
    }

    // Helper method for updating stats based on chart type
    updateStatsForChartType(chartType) {
        if (chartType === 'orders' && this.isStatsProvider) {
            const deliveryPerc = this.calculateDeliveryPercentage();
            this.updateStatDisplay('.delivery-percentage', deliveryPerc !== null ? 
                (deliveryPerc < 1 ? '<1%' : `${Math.round(deliveryPerc)}%`) : null);
        }
        else if (chartType === 'purchasers') {
            const latestPurchasers = this.getLatestValue(this.purchasersData, 'purchasers', 'dacPurchasers');
            this.updateStatDisplay('.purchasers-count', latestPurchasers !== null ? 
                Math.round(latestPurchasers) : null);
        }
        else if (chartType === 'carbonPricing') {
            const carbonPricingCountries = this.carbonPricingData.carbonPricingCountries;
            const latestCountriesCount = carbonPricingCountries[carbonPricingCountries.length - 1];
            this.updateStatDisplay('.carbon-pricing-countries-count', latestCountriesCount !== null ? 
                Math.round(latestCountriesCount) : null);

            const globalEmissionsPct = this.carbonPricingData.globalEmissionsPct;
            const latestEmissionsPct = globalEmissionsPct[globalEmissionsPct.length - 1];
            console.log(`Updating emissions percentage display: ${latestEmissionsPct} → ${Math.round(latestEmissionsPct)}%`);
            this.updateStatDisplay('.global-emissions-coverage', latestEmissionsPct !== null ? 
                `${Math.round(latestEmissionsPct)}%` : null);
        }
    }

    // Use more efficient data processing methods:
    processCSVData(csvText, requiredColumns, transform) {
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        // Check for required columns
        const columnIndices = {};
        for (const column of requiredColumns) {
            columnIndices[column] = headers.indexOf(column);
            if (columnIndices[column] === -1) {
                logger.error(`Required column not found: ${column}`);
                return null;
            }
        }
        
        // Process data rows
        return lines.slice(1)
            .filter(row => row.trim())
            .map(row => {
                const columns = row.split(',');
                return transform(columns, columnIndices);
            })
            .sort(transform.sort || ((a, b) => 0));
    }

    removeLoadingIndicator(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            // Find and remove all loading indicators in this container
            const loadingIndicators = container.querySelectorAll('.chart-loading');
            loadingIndicators.forEach(indicator => indicator.remove());
        }
    }

    // Add this method to the CDRChart class:
    updateLastUpdatedElements(dateStr) {
        const dateToDisplay = dateStr || this.lastUpdated;
        if (!dateToDisplay) return;
        
        this.lastUpdated = dateToDisplay; // Always store the date as-is
        
        // Update all date elements on the page with consistent formatting
        document.querySelectorAll('[data-last-updated]').forEach(el => {
            el.textContent = dateToDisplay;
        });
        
        logger.log(`Updated last updated elements to: ${dateToDisplay}`);
    }
}


/**
 * DealsTicker class handles fetching and displaying latest CDR deal headlines
 * in a scrolling ticker format.
 */
class DealsTicker {
    /**
     * @param {string} containerId - ID of the HTML element to contain the ticker
     * @param {string} apiBaseUrl - Base URL for fetching data
     * @param {Object} options - Configuration options
     */
    constructor(containerId, apiBaseUrl, options = {}) {
        this.containerId = containerId;
        this.apiBaseUrl = apiBaseUrl;
        this.options = {
            refreshInterval: 4 * 60 * 60 * 1000, // 4 hours
            itemCount: 10, // Match CSV row count
            scrollDuration: 120, // Adjust as needed
            dealsKeyPattern: 'processed/latest_deals_*', // Pattern for the deals file
            ...options
        };
        this.dealsUrlPattern = `${this.apiBaseUrl}/presigned-url?key=${this.options.dealsKeyPattern}`;
        this.updateInterval = null;
        this.lastFetchTime = 0;
        this.lastSuccessfulDeals = null;
        this.fetchInProgress = false;
        this.logger = new Logger(); // Use the existing Logger
        injectDealsTickerStyles();
    }

    /**
     * Formats a date string (YYYY-MM-DD HH:MM:SS+ZZ:ZZ) into "MMM dd, yyyy"
     * @param {string} dateString - Date string from CSV
     * @returns {string} Formatted date string
     */
    formatDate(dateString) {
        try {
            // Extract just the date portion (YYYY-MM-DD) and ignore timezone
            const datePart = dateString.substring(0, 10);
            // Parse the date parts manually to avoid timezone issues
            const [year, month, day] = datePart.split('-').map(num => parseInt(num, 10));
            
            // Use array of month names for formatting
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            // Month is 0-indexed in JS, so subtract 1
            const formattedMonth = months[month - 1];
            
            // Return formatted date string
            return `${formattedMonth} ${day}, ${year}`;
        } catch (e) {
            this.logger.error(`Error formatting date: ${dateString}`, e);
            return 'Date Error';
        }
    }

    /**
     * Formats a number into a string with commas.
     * @param {number|string} number - The number to format
     * @returns {string} Formatted number string
     */
    formatTons(number) {
        const num = parseFloat(number);
        if (isNaN(num)) {
            return number; // Return original if not a number
        }
        return num.toLocaleString('en-US');
    }

    /**
     * Fetches latest deals from the CSV via presigned URL
     * @returns {Promise<Array>} Array of deal items
     */
    async fetchDeals() {
        const now = Date.now();
        // Avoid fetching too frequently or while another fetch is in progress
        if (this.fetchInProgress || (now - this.lastFetchTime < 60000 && this.lastSuccessfulDeals)) {
            this.logger.debug('Deals fetch skipped (too recent or in progress)');
            return this.lastSuccessfulDeals || []; // Return last known deals or empty
        }

        this.fetchInProgress = true;
        this.lastFetchTime = now;
        this.logger.log('Fetching latest deals...');

        try {
            // 1. Get the presigned URL for the latest deals file
            const presignedResponse = await fetch(this.dealsUrlPattern);
            if (!presignedResponse.ok) {
                throw new Error(`Failed to get presigned URL: ${presignedResponse.statusText}`);
            }
            const presignedData = await presignedResponse.json();
            if (!presignedData.url) {
                throw new Error('No presigned URL returned from API for deals');
            }
            const actualCsvUrl = presignedData.url;
            this.logger.debug(`Got presigned URL for deals: ${actualCsvUrl.substring(0, 100)}...`);

            // 2. Fetch the actual CSV data
            const csvResponse = await fetch(actualCsvUrl);
            if (!csvResponse.ok) {
                throw new Error(`Failed to fetch deals CSV: ${csvResponse.statusText}`);
            }
            const csvText = await csvResponse.text();

            // 3. Parse the CSV data
            const lines = csvText.split('\n').filter(row => row.trim());
            if (lines.length < 2) { // Need header + at least one data row
                 throw new Error('CSV data is empty or missing header');
            }
            const headers = lines[0].split(',').map(h => h.trim());
            const deals = lines.slice(1).map(line => {
                const values = line.split(',');
                const deal = {};
                headers.forEach((header, index) => {
                    deal[header] = values[index] ? values[index].trim() : '';
                });
                return deal;
            });

            // 4. Sort by announcement_date (descending - most recent first)
            // Ensure robust date parsing for sorting
            deals.sort((a, b) => {
                const dateA = new Date(a.announcement_date).getTime();
                const dateB = new Date(b.announcement_date).getTime();
                // Handle potential invalid dates during sort
                if (isNaN(dateA) && isNaN(dateB)) return 0;
                if (isNaN(dateA)) return 1; // Put invalid dates last
                if (isNaN(dateB)) return -1; // Put invalid dates last
                return dateB - dateA; // Descending order
            });


            this.lastSuccessfulDeals = deals.slice(0, this.options.itemCount); // Store the latest deals
            this.logger.log(`Successfully fetched ${this.lastSuccessfulDeals.length} deals.`);
            return this.lastSuccessfulDeals;

        } catch (error) {
            this.logger.error('Deals fetch error:', error);
            // Optionally return fallback data or just the last successful fetch
            return this.lastSuccessfulDeals || [];
        } finally {
            this.fetchInProgress = false;
        }
    }

    /**
     * Renders deal items in the ticker container
     * @param {Array} dealItems - Array of deal items to display
     */
    renderTicker(dealItems) {
        const container = document.getElementById(this.containerId);
        
        if (!container) {
            this.logger.error(`Deals ticker container #${this.containerId} not found.`);
            return;
        }

        if (!dealItems || dealItems.length === 0) {
            this.logger.warn('No deal items to display in ticker.');
            container.innerHTML = '<div class="deals-ticker-content" style="padding: 10px;">No recent deals data available.</div>';
            return;
        }

        // Find column indices robustly
        const sampleDeal = dealItems[0];
        
        const dateKey = Object.keys(sampleDeal).find(k => k.toLowerCase().includes('date'));
        const purchaserKey = Object.keys(sampleDeal).find(k => k.toLowerCase().includes('purchaser'));
        const tonsKey = Object.keys(sampleDeal).find(k => k.toLowerCase().includes('tons'));
        const supplierKey = Object.keys(sampleDeal).find(k => k.toLowerCase().includes('supplier'));
        const methodKey = Object.keys(sampleDeal).find(k => k.toLowerCase().includes('method'));

        if (!dateKey || !purchaserKey || !tonsKey || !supplierKey || !methodKey) {
            this.logger.error('Could not find required keys in deal data:', Object.keys(sampleDeal));
            container.innerHTML = '<div class="deals-ticker-content" style="padding: 10px;">Error processing deal data format.</div>';
            return;
        }

        const content = dealItems.map(item => {
            const date = this.formatDate(item[dateKey]);
            const purchaser = item[purchaserKey] || 'Unknown Purchaser';
            const tons = this.formatTons(item[tonsKey] || 0);
            const supplier = item[supplierKey] || 'Unknown Supplier';
            const method = item[methodKey] || 'Unknown Method';
            
            // Check if this is a DAC deal
            const isDacDeal = method.toLowerCase().includes('dac');
            
            // Add "New DAC Deal" prefix with green color for DAC methods
            const dacPrefix = isDacDeal ? 
                `<span style="color: #20DD7B; font-weight: bold;">${this.getTranslation('NEW DAC DEAL')}</span>&nbsp;&nbsp;` : 
                '';
            
            // Format the ticker item with conditional DAC prefix
            const tickerItem = `<span class="deal-item">${dacPrefix}${date} | ${purchaser} ${this.getTranslation('purchased')} ${tons} ${this.getTranslation('tons from')} ${supplier} ${this.getTranslation('with')} ${method}.</span>`;
            return tickerItem;
        }).join('        '); // Separator between items

        // Duplicate content for seamless looping animation
        const finalHTML = `
            <div class="deals-ticker-content" style="animation: dealsTickerAnim ${this.options.scrollDuration}s linear infinite">
                ${content}${content}
            </div>
        `;
        container.innerHTML = finalHTML;

        this.logger.log('Deals ticker content rendered.');
    }

    getTranslation(englishText) {
        const isFrench = isLanguageFrench();
        if (!isFrench) return englishText;
        
        const translations = {
            'TONS PURCHASED': 'TONNES ACHETÉES',
            'TONS DELIVERED': 'TONNES LIVRÉES',
            'CUMULATIVE UNIQUE PURCHASERS': 'ACHETEURS UNIQUES CUMULÉS',
            'COUNTRIES WITH CARBON PRICING': 'PAYS AVEC TARIFICATION DU CARBONE',
            'TONS': 'TONNES',
            'K TONS': 'K TONNES',
            'M TONS': 'M TONNES'
        };
        
        return translations[englishText] || englishText;
    }

    async update() {
        this.logger.debug('Updating deals ticker...');
        const deals = await this.fetchDeals();
        this.renderTicker(deals);
    }

    /**
     * Starts the deals ticker
     */
    start() {
        this.logger.log('Starting deals ticker...');
        this.update(); // Initial fetch and render
        if (this.options.refreshInterval > 0) {
            this.updateInterval = setInterval(() => this.update(), this.options.refreshInterval);
        }

        // Pause animation when tab is not visible
        document.addEventListener('visibilitychange', () => {
            const tickerContent = document.querySelector(`#${this.containerId} .deals-ticker-content`);
            if (tickerContent) {
                tickerContent.style.animationPlayState = document.hidden ? 'paused' : 'running';
            }
            // Also refresh when the tab becomes visible again if it's been a while
            if (!document.hidden && Date.now() - this.lastFetchTime > 30 * 60 * 1000) { // e.g., refresh if hidden > 30 mins
                this.logger.log('Tab visible after period, refreshing deals ticker.');
                this.update();
            }
        });
    }

    /**
     * Stops the deals ticker and cleans up resources
     */
    stop() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = ''; // Clear content
        }
        // Consider removing visibilitychange listener if needed, though usually harmless
        this.logger.log('Deals ticker stopped.');
    }
}

if (typeof window !== 'undefined') {
    window.CDRChart = CDRChart;
    window.DealsTicker = DealsTicker;
    window.CDRStateManager = CDRStateManager;
    window.findLatestDateSuffixedFile = findLatestDateSuffixedFile;
}