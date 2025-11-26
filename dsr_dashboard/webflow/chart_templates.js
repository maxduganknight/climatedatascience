// Logger class for managing logging behavior
class Logger {
    constructor(isProduction = false) {
        this.isProduction = isProduction;
        
        // Determine environment
        this.isProduction = window.location.hostname.includes('deepskyclimate.com') ||
                           window.location.hostname.includes('webflow.io');
    }

    log(...args) {
        if (!this.isProduction) {
            console.log(...args);
        }
    }

    error(...args) {
        // Always log errors, but in production only log to error monitoring service if set up
        if (this.isProduction) {
            // Here you could add error monitoring service integration
            // e.g., Sentry, LogRocket, etc.
        } else {
            console.error(...args);
        }
    }

    debug(...args) {
        if (!this.isProduction) {
            console.debug(...args);
        }
    }
}

// Create global logger instance
const logger = new Logger();

// Constants
const COLORS = {
    RED: '#EC6252',
    GREY_2024: '#929190',
    GREY_2023: '#505151',
    LIGHT_GREY: '#64656550'
};

const FONTS = {
    FAMILY: "'Space Mono', monospace",
    SIZE: 12,
    COLOR: '#918F90',
    STYLE: (text) => text.toUpperCase()
};

const search_words = [
    'climate',
    'wildfire',
    'sea level',
    'storm',
    'hurricane',
    'heat wave',
    'severe weather',
    'extreme weather',
    'drought',
    'flooding'
]
const exclude_words = [
    'trump',
    'musk'
]
const RSS_URL = `https://news.google.com/rss/search?q=intitle%3A(${search_words.map(word => `%22${word.replace(/ /g, '%20')}%22`).join('%20OR%20')})${exclude_words.map(word => `%20-%22${word}%22`).join('')}%20when%3A3d&hl=en-US&gl=US&ceid=US%3Aen`

// // 1. First, add this simple legend alignment plugin near the top of your file (after constants)
// // Simple plugin to align legend with the left edge of the chart
// const legendAlignPlugin = {
//     id: 'legendAlignPlugin',
//     beforeLayout: function(chart) {
//         // Only apply to charts with legends aligned to start
//         if (chart.options.plugins.legend && 
//             chart.options.plugins.legend.display &&
//             chart.options.plugins.legend.align === 'start') {
            
//             // Add negative margin to pull the legend left
//             if (!chart.options.layout) chart.options.layout = {};
//             if (!chart.options.layout.padding) chart.options.layout.padding = {};
            
//             chart.options.layout.padding.left = -7.5;
//         }
//     }
// };

// // Register plugin globally
// if (typeof Chart !== 'undefined') {
//     Chart.register(legendAlignPlugin);
// }

// Shared chart configurations
const baseChartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
        padding: {
            left: 0,
            right: 0
        }
    },
    plugins: {
        legend: { display: false },
        title: { display: false }
    }
};

const createAxisConfig = (isYAxis = false) => ({
    grid: {
        display: isYAxis,
        color: isYAxis ? 'rgba(128, 128, 128, 0.25)' : undefined
    },
    border: {
        display: !isYAxis
    },
    position: isYAxis ? 'right' : undefined,
    title: { display: false },
    ticks: {
        color: FONTS.COLOR,
        font: {
            family: FONTS.FAMILY,
            size: FONTS.SIZE
        },
        maxTicksLimit: isYAxis ? undefined : 6,
        autoSkip: isYAxis ? undefined : true,
        align: isYAxis ? 'center' : 'start',
        padding: 8,
        callback: function(value) {
            const label = this.getLabelForValue(value);
            return isYAxis ? FONTS.STYLE(label) : FONTS.STYLE(label.split('-')[0]);
        }
    }
    // ,
    // // Eliminate padding
    // afterFit: function(axis) {
    //     if (!isYAxis) {
    //         // Remove padding for x-axis
    //         axis.paddingLeft = 0;
    //         axis.paddingRight = 0;
    //     }
    // }
});

const createTooltipConfig = (yAxisUnit = '', isDailyLine = false) => ({
    displayColors: false,
    intersect: isDailyLine,
    titleFont: { family: FONTS.FAMILY, size: FONTS.SIZE },
    bodyFont: { family: FONTS.FAMILY, size: FONTS.SIZE },
    callbacks: {
        title: context => {
            if (!context.length) return '';
            if (isDailyLine) {
                const date = context[0].raw.x;
                const currentYear = new Date().getFullYear();
                const [month, day] = date.split('-');
                return FONTS.STYLE(`${currentYear}-${month}-${day}`);
            }
            return FONTS.STYLE(context[0].label);
        },
        label: context => {
            if (!context.raw) return '';
            if (isDailyLine) {
                // Find the nearest visible dataset
                const nearestPoint = context.chart.getElementsAtEventForMode(
                    context.chart.canvas.getBoundingClientRect(), 
                    'nearest', 
                    { intersect: true },
                    false
                )[0];
                
                if (nearestPoint && nearestPoint.datasetIndex === context.datasetIndex) {
                    const value = Number(context.raw.y);
                    return FONTS.STYLE(`${value.toFixed(2)} ${yAxisUnit}`);
                }
                return null;
            }
            const value = Number(context.raw);
            return FONTS.STYLE(`${value.toFixed(2)} ${yAxisUnit}`);
        }
    }
});

// Chart templates with shared configurations
const chartTemplates = {
    bar: {
        type: 'bar',
        defaultOptions: (yAxisUnit = '') => ({
            ...baseChartConfig,
            scales: {
                x: createAxisConfig(false),
                y: createAxisConfig(true)
            },
            plugins: {
                ...baseChartConfig.plugins,
                tooltip: {
                    ...createTooltipConfig(yAxisUnit),
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false,
                    position: 'nearest'
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            barPercentage: 0.98,
            categoryPercentage: 0.8
        })
    },
    line: {
        type: 'line',
        defaultOptions: (yAxisUnit = '') => ({
            ...baseChartConfig,
            scales: {
                x: createAxisConfig(false),
                y: createAxisConfig(true)
            },
            plugins: {
                ...baseChartConfig.plugins,
                tooltip: createTooltipConfig(yAxisUnit)
            }
        })
    },
    daily_line: {
        type: 'line',
        defaultOptions: (yAxisUnit = '') => ({
            ...baseChartConfig,
            elements: {
                line: { tension: 0.3, borderWidth: 2 },
                point: { 
                    radius: 0,
                    hoverRadius: 18
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                x: {
                    ...createAxisConfig(false),
                    type: 'time',
                    time: {
                        unit: 'month',
                        parser: 'MM-dd',
                        displayFormats: { 
                            month: 'MMM'
                        }
                    },
                    ticks: {
                        color: FONTS.COLOR,
                        font: {
                            family: FONTS.FAMILY,
                            size: FONTS.SIZE
                        },
                        maxTicksLimit: 6,
                        autoSkip: true,
                        align: 'start',
                        // Force horizontal labels
                        minRotation: 0,
                        maxRotation: 0,
                        callback: function(value, index, values) {
                            // Get month number (0-11) from the timestamp
                            const date = new Date(value);
                            const monthIndex = date.getMonth();
                            
                            // Use Month abbreviation based on language
                            const monthsShort = isLanguageFrench()
                                ? ['JAN', 'FÉV', 'MAR', 'AVR', 'MAI', 'JUIN', 'JUIL', 'AOÛ', 'SEP', 'OCT', 'NOV', 'DÉC']
                                : ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
                                
                            return monthsShort[monthIndex];
                        }
                    }
                },
                y: createAxisConfig(true)
            },
            plugins: {
                ...baseChartConfig.plugins,
                tooltip: {
                    enabled: true,
                    mode: 'nearest',
                    intersect: true,
                    displayColors: false,
                    filter: (tooltipItem) => {
                        const lastIndex = tooltipItem.chart.data.datasets.length - 1;
                        return tooltipItem.datasetIndex >= lastIndex - 3 && tooltipItem.datasetIndex < lastIndex;
                    },
                    callbacks: {
                        title: context => {
                            if (!context.length) return '';
                            const date = context[0].raw.x;
                            const [month, day] = date.split('-');
                            const year = context[0].dataset.label;
                            return FONTS.STYLE(`${year}-${month}-${day}`);
                        },
                        label: context => {
                            if (!context.raw) return '';
                            const value = Number(context.raw.y);
                            return FONTS.STYLE(`${value.toFixed(2)} ${yAxisUnit}`);
                        }
                    }
                },
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        font: {
                            family: FONTS.FAMILY,
                            size: FONTS.SIZE
                        },
                        pointStyle: 'line',
                        boxWidth: 20,
                        boxHeight: 2,
                        padding: 10,
                        filter: function(legendItem, data) {
                            if (!legendItem || legendItem.datasetIndex === undefined) {
                                return false;
                            }
                            return legendItem.datasetIndex >= data.datasets.length - 4;
                        },
                        generateLabels: (chart) => {
                            if (!chart || !chart.data || !chart.data.datasets) {
                                return [];
                            }
                            const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                            return labels.map(label => ({
                                ...label,
                                text: FONTS.STYLE(label.text),
                                lineWidth: 4
                            }));
                        }
                    },
                    display: true,
                    onClick: null
                }
            },
            events: ['mousemove', 'mouseout', 'touchstart', 'touchmove']
        })
    }
};

/**
 * NewsTicker class handles fetching and displaying climate-related news headlines
 * in a scrolling ticker format.
 */
class NewsTicker {
    /**
     * @param {string} containerId - ID of the HTML element to contain the ticker
     * @param {Object} options - Configuration options
     */
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            refreshInterval: 6 * 60 * 60 * 1000, // 6 hours
            itemCount: 10,
            scrollDuration: 120,
            ...options
        };
        this.apiBaseUrl = options.apiBaseUrl;
        this.newsUrl = RSS_URL;
        this.updateInterval = null;
        this.lastFetchTime = 0;
        this.lastSuccessfulFetch = null;
        this.fetchInProgress = false;
    }

    /**
     * Formats a date string into a readable format
     * @param {string} dateString - Date string from RSS feed
     * @returns {string} Formatted date string
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: 'numeric' 
        });
    }

    /**
     * Fetches news items from Google News RSS feed via Lambda function
     * @returns {Promise<Array>} Array of news items
     */
    async fetchNews() {
        // Don't fetch too frequently
        const now = Date.now();
        if (this.fetchInProgress || (now - this.lastFetchTime < 60000 && this.lastSuccessfulFetch)) {
            return this.lastSuccessfulFetch || this.getFallbackNews();
        }
        
        this.fetchInProgress = true;
        this.lastFetchTime = now;
        
        try {
            if (!this.apiBaseUrl) {
                throw new Error('No API base URL configured');
            }
            
            const encodedRssUrl = encodeURIComponent(this.newsUrl);
            const rssEndpoint = `${this.apiBaseUrl}/rss-feed?url=${encodedRssUrl}`;
            
            const response = await fetch(rssEndpoint);
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.items || !data.items.length) {
                throw new Error('No news items returned');
            }
            
            // Cache the successful result
            this.lastSuccessfulFetch = data.items;
            return data.items;
            
        } catch (error) {
            console.error('News fetch error:', error.message);
            return this.lastSuccessfulFetch || this.getFallbackNews();
        } finally {
            this.fetchInProgress = false;
        }
    }

    getFallbackNews() {
        console.log('Using fallback news items');
        return [
            { 
                title: "Climate disasters lead to billions in insurance losses. Could they trigger a financial crisis?", 
                source: "CBC News", 
                date: "Feb 05, 2025" 
            },
            { 
                title: "Europe 'can't cope' with extreme weather costs, warns insurance watchdog", 
                source: "Financial Times", 
                date: "Feb 03, 2025" 
            },
            { 
                title: "Climate Driven Population Shifts and Insurance Increases are set to Erase $1.4 Trillion in American Real Estate Value", 
                source: "First Street", 
                date: "Feb 03, 2025" 
            },
            { 
                title: "Climate change increases threat of heat deaths in European cities", 
                source: "Financial Times", 
                date: "Jan 27, 2025" 
            },
            { 
                title: "Insurers Are Deserting Homeowners as Climate Shocks Worsen", 
                source: "The New York Times", 
                date: "Dec 18, 2024" 
            }
        ];
    }

    /**
     * Renders news items in the ticker container
     * @param {Array} newsItems - Array of news items to display
     */
    renderTicker(newsItems) {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`News ticker container #${this.containerId} not found in the DOM`);
            return;
        }
        
        if (!newsItems || !newsItems.length) {
            console.warn('No news items to display in ticker');
            return;
        }        
        const content = newsItems.map(item => 
            `<span class="news-item">${item.source} • ${item.date} | ${item.title}</span>`
        ).join('        ');

        container.innerHTML = `
            <div class="ticker-content" style="animation: ticker ${this.options.scrollDuration}s linear infinite">
                ${content}${content}
            </div>
        `;
        
        console.log('News ticker content rendered');
    }

    /**
     * Updates the ticker with fresh news
     */
    async update() {
        const news = await this.fetchNews();
        this.renderTicker(news);
    }

    /**
     * Starts the news ticker
     */
    start() {
        this.update();
        if (this.options.refreshInterval) {
            this.updateInterval = setInterval(() => this.update(), this.options.refreshInterval);
        }

        // Pause animation when tab is not visible
        document.addEventListener('visibilitychange', () => {
            const ticker = document.querySelector('.ticker-content');
            if (ticker) {
                ticker.style.animationPlayState = document.hidden ? 'paused' : 'running';
            }
        });

        // Also refresh when the tab becomes visible again if it's been a while
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && Date.now() - this.lastFetchTime > 30 * 60 * 1000) {
                this.update();
            }
        });
    }

    /**
     * Stops the news ticker and cleans up resources
     */
    stop() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = '';
        }
    }
}

/**
 * DeepSkyCharts class manages the creation and updating of climate data charts
 */
class DeepSkyCharts {
    /**
     * @param {string} baseUrl - Base URL for data fetching
     * @param {Object} options - Configuration options
     * @param {boolean} [options.is_local=false] - Whether to use local data
     */
    constructor(baseUrl, options = {}) {
        console.log('DeepSkyCharts constructor called with:', {
            baseUrl,
            options
        });

        this.is_local = options.is_local || false;
        this.DATA_PATH = this.is_local ? baseUrl : null;
        
        // Accept both camelCase and snake_case versions with better logging
        this.API_BASE_URL = options.apiBaseUrl || options.api_base_url || (this.is_local ? null : baseUrl);
                
        this.charts = new Map();
        this.latestValues = new Map();
        this.lastUpdates = new Map();
        this.cache = new Map();
        this.cacheExpiry = new Map();
        this.CACHE_DURATION = 1000 * 60 * 60 * 24; // 24 hours
        this.lastVersionCheck = 0;
        this.VERSION_CHECK_INTERVAL = 1000 * 60 * 60 * 24; // Check versions once per day
        this.forceVersionCheck = false;

        // Initialize news ticker with the API base URL from parent class
        this.newsTicker = new NewsTicker('newsTicker', {
            refreshInterval: 6 * 60 * 60 * 1000,
            itemCount: 10,
            apiBaseUrl: this.API_BASE_URL // Pass API base URL here
        });
        this.newsTicker.start();

        // Add formatters configuration
        this.formatters = {
            'co2_ppm': value => Math.round(value).toString(),
            'aviso_slr': value => Math.round(value).toString(),
            'arctic_sea_ice': value => value.toFixed(2),
            'home_insurance_premium': value => Math.round(value).toString(),
            'noaa_billion': value => Math.round(value).toString(),
            'era5_t2m_anom_year': value => value.toFixed(2),
            'era5_sst_anom_month_day': value => value.toFixed(2),
            'era5_t2m_anom_month_day': value => value.toFixed(2)
        };
    }

    async getPresignedUrl(filename) {
        // Check cache first
        const cacheKey = `presigned_${filename}`;
        const cached = this.cache.get(cacheKey);
        const expiry = this.cacheExpiry.get(cacheKey);
        
        if (cached && expiry && Date.now() < expiry) {
            return cached;
        }

        if (this.is_local) {
            const basePattern = filename.replace('_*.csv', '_');
            try {
                const response = await fetch(`${this.DATA_PATH}?pattern=${basePattern}`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const files = await response.json();
                const mostRecentFile = files.sort().pop();
                if (!mostRecentFile) throw new Error(`No matching file found for pattern: ${filename}`);
                const url = `${this.DATA_PATH}/${mostRecentFile}`;
                
                // Cache the result
                this.cache.set(cacheKey, url);
                this.cacheExpiry.set(cacheKey, Date.now() + this.CACHE_DURATION);
                return url;
            } catch (error) {
                logger.error('Error finding local file:', error);
                throw error;
            }
        }

        try {
            logger.debug('Fetching presigned URL for:', filename);
            const response = await fetch(
                `${this.API_BASE_URL}/presigned-url?key=${filename}`,
                { headers: { 'Origin': 'https://www.deepskyclimate.com' } }
            );
            const data = await response.json();
            
            // Cache the result
            this.cache.set(cacheKey, data.url);
            this.cacheExpiry.set(cacheKey, Date.now() + this.CACHE_DURATION);
            return data.url;
        } catch (error) {
            logger.error('Error getting presigned URL:', error);
            throw error;
        }
    }
    
    async fetchData(filename, x_var, y_var, skipVersionCheck = false) {
        // Check for new versions only if necessary and not skipped
        if (!skipVersionCheck && Date.now() - this.lastVersionCheck >= this.VERSION_CHECK_INTERVAL) {
            await this.checkForNewVersions();
        }
        
        // Check cache first 
        const cacheKey = `data_${filename}_${x_var}_${y_var}`;
        const cached = this.cache.get(cacheKey);
        const expiry = this.cacheExpiry.get(cacheKey);
        
        if (cached && expiry && Date.now() < expiry) {
            logger.debug('Using cached data for:', filename);
            return cached;
        }

        try {
            const url = await this.getPresignedUrl(filename);
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.text();
            const rows = data.split('\n').filter(row => row.trim());
            const header = rows[0].split(',');
            const x_index = header.indexOf(x_var);
            const y_index = header.indexOf(y_var);
            
            if (x_index === -1 || y_index === -1) {
                throw new Error(`Could not find columns: ${x_var}, ${y_var}`);
            }
            
            const labels = [];
            const values = [];
            
            for (let i = 1; i < rows.length; i++) {
                const cols = rows[i].split(',');
                labels.push(cols[x_index]);
                values.push(parseFloat(cols[y_index]));
            }
            
            const result = { labels, values };

            // Cache the result
            this.cache.set(cacheKey, result);
            this.cacheExpiry.set(cacheKey, Date.now() + this.CACHE_DURATION);
            return result;
        } catch (error) {
            logger.error("Error fetching data:", error);
            return { labels: [], values: [] };
        }
    }

    async fetchDailyData(filename, skipVersionCheck = false) {
        // Check for new versions only if necessary and not skipped
        if (!skipVersionCheck && Date.now() - this.lastVersionCheck >= this.VERSION_CHECK_INTERVAL) {
            await this.checkForNewVersions();
        }

        // Check cache first
        const cacheKey = `daily_${filename}`;
        const cached = this.cache.get(cacheKey);
        const expiry = this.cacheExpiry.get(cacheKey);
        
        if (cached && expiry && Date.now() < expiry) {
            return cached;
        }

        try {
            const url = await this.getPresignedUrl(filename);
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.text();
            const rows = data.split('\n').filter(row => row.trim());
            const headers = rows[0].split(',').map(h => h.trim());
            const years = headers.slice(1);
            const latestYearIndex = years.length;
            
            let latestValue = null;
            let latestDate = null;
            
            for (let i = rows.length - 1; i > 0; i--) {
                const columns = rows[i].split(',').map(col => col.trim());
                const value = parseFloat(columns[latestYearIndex]);
                if (!isNaN(value)) {
                    latestValue = value;
                    latestDate = columns[0];
                    break;
                }
            }

            let datasets = years.map((year, index) => {
                const values = rows.slice(1)
                    .filter(row => {
                        const dayMonth = row.split(',')[0].trim();
                        return dayMonth !== '29-Feb';
                    })
                    .map(row => {
                        const columns = row.split(',').map(col => col.trim());
                        return {
                            x: columns[0],
                            y: parseFloat(columns[index + 1])
                        };
                    });

                const lastIndex = years.length - 1;
                const isRecentYear = index >= lastIndex - 2;

                const borderColor = index === lastIndex ? COLORS.RED :         
                                  index === lastIndex - 1 ? COLORS.GREY_2024 :      
                                  index === lastIndex - 2 ? COLORS.GREY_2023 : 
                                  COLORS.LIGHT_GREY;                           

                const borderWidth = index === lastIndex ? 4 :           
                                  isRecentYear ? 2 : .65;     

                return {
                    label: year,
                    data: values,
                    borderColor,
                    borderWidth,
                    fill: false,
                    hidden: false,
                    pointRadius: 0,
                    pointHoverRadius: isRecentYear ? 12 : 0
                };
            });

            const historicalDataset = {
                label: '79-22',
                data: [],
                borderColor: '#323232',
                borderWidth: 4,
                fill: false,
                hidden: false,
                pointRadius: 0,
                pointHoverRadius: 0
            };

            const lastThree = datasets.slice(-3).reverse();
            const others = datasets.slice(0, -3);
            datasets = [...others, ...lastThree, historicalDataset];
            
            const result = {
                datasets,
                latestValue: latestValue ? { value: latestValue, date: latestDate } : null
            };

            // Cache the result
            this.cache.set(cacheKey, result);
            this.cacheExpiry.set(cacheKey, Date.now() + this.CACHE_DURATION);
            return result;
        } catch (error) {
            console.error("Error fetching daily data:", error);
            return { datasets: [] };
        }
    }

    async initializeChart(config, prefetchedUrl, prefetchedData = null) {
        const { canvasId, type = 'bar', filename, x_var, y_var, y_axis_unit = '' } = config;

        if (!chartTemplates[type]) {
            throw new Error(`Unknown chart type: ${type}`);
        }

        try {
            const canvas = document.getElementById(canvasId);
            if (!canvas) throw new Error(`Canvas element not found: ${canvasId}`);
            
            // Use pre-fetched URL instead of fetching again
            const url = prefetchedUrl || await this.getPresignedUrl(filename);
            
            // Extract date from either local path or S3 URL
            let dateStr = null;
            if (this.is_local) {
                const match = url.match(/\d{8}\.csv$/);
                if (match) {
                    dateStr = match[0].slice(0, 8);
                }
            } else {
                // For S3 URLs, look for the date pattern anywhere in the URL
                const dateMatch = url.match(/(\d{8})\.csv/);
                if (dateMatch) {
                    dateStr = dateMatch[1];  // Use capture group to get just the numbers
                }
            }
                        
            if (dateStr) {
                this.lastUpdates.set(canvasId, dateStr);
                // Update the last update text immediately
                const updateElement = document.getElementById(`${canvasId}_last_update`);
                if (updateElement) {
                    const formattedDate = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
                    updateElement.textContent = formattedDate;
                }
            }

            const chartData = type === 'daily_line' ?
                await this.fetchDailyData(filename) :
                prefetchedData || await this.fetchData(filename, x_var, y_var);

            if (type === 'daily_line') {
                if (chartData.latestValue) {
                    this.latestValues.set(canvasId, chartData.latestValue);
                }
            } else if (chartData.values?.length > 0) {
                this.latestValues.set(canvasId, {
                    value: chartData.values[chartData.values.length - 1],
                    date: chartData.labels[chartData.labels.length - 1]
                });
            }

            const template = chartTemplates[type];
            const chartConfig = this.buildChartConfig(type, template, chartData, y_axis_unit);

            const chart = new Chart(canvas.getContext('2d'), chartConfig);
            this.charts.set(canvasId, chart);
            
            // Add style to remove container padding
            const container = canvas.parentElement;
            if (container) {
                container.style.padding = '0';
                container.style.margin = '0';
            }
        } catch (error) {
            console.error(`Error initializing chart ${canvasId}:`, error);
        }
    }

    buildChartConfig(type, template, chartData, yAxisUnit = '') {
        // For daily_line charts
        if (type === 'daily_line') {
            // Make sure chartData has valid datasets to prevent null errors
            const datasets = chartData && chartData.datasets ? chartData.datasets : [];
            
            // Safety check to ensure all datasets have proper structure
            const validatedDatasets = datasets.map(dataset => ({
                ...dataset,
                data: dataset.data || [],  // Ensure data exists
                hidden: false,             // Explicitly set hidden property
            }));
            
            return {
                type: 'line',
                data: {
                    datasets: validatedDatasets
                },
                options: template.defaultOptions(yAxisUnit)
            };
        } 
        // For other chart types
        else {
            return {
                type: template.type,
                data: {
                    labels: chartData.labels || [],
                    datasets: [{
                        label: '',
                        data: chartData.values || [],
                        backgroundColor: type === 'bar' ? COLORS.RED : 'transparent',
                        borderColor: COLORS.RED,
                        borderWidth: type === 'line' ? 2 : 0,
                        tension: 0.3,
                        pointRadius: 0,
                        borderRadius: type === 'bar' ? 5 : 0,
                        hoverBackgroundColor: type === 'bar' ? '#FFFFFF' : undefined
                    }]
                },
                options: template.defaultOptions(yAxisUnit)
            };
        }
    }

    getLatestValue(chartId) {
        const latestValue = this.latestValues.get(chartId);
        if (!latestValue || latestValue.value === undefined) return null;

        const formatter = this.formatters[chartId];
        return {
            ...latestValue,
            formattedValue: formatter ? formatter(latestValue.value) : `${latestValue.value}`
        };
    }

    getLastUpdate(chartId) {
        const dateStr = this.lastUpdates.get(chartId);
        if (!dateStr) return null;
        
        // Format as YYYY-MM-DD
        return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
    }

    // Enhance the batchFetchData method to be even more efficient
    async batchFetchData(requests) {
        const results = new Map();
        
        // Group requests by filename to minimize redundant fetches
        const fileGroups = new Map();
        requests.forEach(({ filename, x_var, y_var }) => {
            if (!fileGroups.has(filename)) {
                fileGroups.set(filename, []);
            }
            fileGroups.get(filename).push({ x_var, y_var });
        });
        
        // Fetch all files in parallel
        const filePromises = Array.from(fileGroups).map(async ([filename, vars]) => {
            try {
                const url = await this.getPresignedUrl(filename);
                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const data = await response.text();
                const rows = data.split('\n').filter(row => row.trim());
                const header = rows[0].split(',');
                
                vars.forEach(({ x_var, y_var }) => {
                    const x_index = header.indexOf(x_var);
                    const y_index = header.indexOf(y_var);
                    
                    if (x_index === -1 || y_index === -1) {
                        throw new Error(`Could not find columns: ${x_var}, ${y_var}`);
                    }
                    
                    const values = rows.slice(1).map(row => {
                        const cols = row.split(',');
                        return {
                            x: cols[x_index],
                            y: parseFloat(cols[y_index])
                        };
                    });
                    
                    results.set(`${filename}_${x_var}_${y_var}`, {
                        labels: values.map(v => v.x),
                        values: values.map(v => v.y)
                    });
                });
            } catch (error) {
                logger.error(`Error processing file ${filename}:`, error);
            }
        });
        
        // Wait for all file processing to complete
        await Promise.all(filePromises);
        
        return results;
    }

    // Add this method to preload data before charts are initialized
    async preloadData(configArray) {
        try {
            // Group configs by chart type
            const standardCharts = configArray.filter(config => config.type !== 'daily_line');
            const dailyCharts = configArray.filter(config => config.type === 'daily_line');
            
            const dataPromises = [];
            
            // Preload standard chart data
            if (standardCharts.length > 0) {
                const batchRequests = standardCharts.map(config => ({
                    filename: config.filename,
                    x_var: config.x_var,
                    y_var: config.y_var
                }));
                
                dataPromises.push(this.batchFetchData(batchRequests));
            }
            
            // Preload daily chart data in parallel
            dailyCharts.forEach(config => {
                dataPromises.push(
                    this.fetchDailyData(config.filename)
                        .then(data => {
                            // Cache the result with a specific key for daily data
                            const cacheKey = `daily_${config.filename}`;
                            this.cache.set(cacheKey, data);
                            this.cacheExpiry.set(cacheKey, Date.now() + this.CACHE_DURATION);
                            return data;
                        })
                );
            });
            
            // Wait for all data to be preloaded
            await Promise.all(dataPromises);
            logger.log('All chart data preloaded');
            
        } catch (error) {
            logger.error('Error preloading data:', error);
        }
    }

    async checkForNewVersions() {
        // Only check once per day or when forced
        const now = Date.now();
        if (!this.forceVersionCheck && now - this.lastVersionCheck < this.VERSION_CHECK_INTERVAL) {
            logger.debug('Skipping version check - checked recently');
            return false;
        }
        
        this.forceVersionCheck = false;
        
        try {
            logger.log('Checking for new dataset versions...');
            
            const versionsUrl = this.is_local 
                ? `${this.DATA_PATH.split('/data/processed')[0]}/dataset-versions`
                : `${this.API_BASE_URL}/dataset-versions`;
                
            const response = await fetch(versionsUrl, {
                headers: { 'Origin': 'https://www.deepskyclimate.com' },
                cache: 'no-store' // Always get fresh version info
            });
            
            if (!response.ok) {
                if (this.is_local) {
                    logger.log('Local development: Skipping version check');
                    this.lastVersionCheck = now;
                    return false;
                }
                throw new Error(`Failed to fetch dataset versions: ${response.status}`);
            }
            
            const data = await response.json();
            let hasNewVersions = false;
            
            if (!data.versions) {
                logger.error('Invalid version response format:', data);
                this.lastVersionCheck = now;
                return false;
            }
            
            for (const [filename, version] of Object.entries(data.versions)) {
                const cacheKey = `presigned_${filename}`;
                const cachedVersion = this.getVersionFromUrl(this.cache.get(cacheKey));
                
                // If there's a newer version or we don't have this version cached
                if (!cachedVersion || version > cachedVersion) {
                    if (cachedVersion) {
                        logger.log(`New version detected for ${filename}: ${cachedVersion} -> ${version}`);
                    } else {
                        logger.log(`First time caching ${filename}: ${version}`);
                    }
                    
                    // Invalidate any cache entries related to this file
                    this.invalidateFileCaches(filename);
                    hasNewVersions = true;
                }
            }
            
            this.lastVersionCheck = now;
            return hasNewVersions;
        } catch (error) {
            logger.error('Error checking for new versions:', error);
            this.lastVersionCheck = now;
            return false;
        }
    }

    // New helper method to consolidate cache invalidation
    invalidateFileCaches(filename) {
        // Remove the presigned URL cache
        const cacheKey = `presigned_${filename}`;
        this.cache.delete(cacheKey);
        this.cacheExpiry.delete(cacheKey);
        
        // Remove any data cache entries for this file
        const baseFilename = filename.replace('_*.csv', '');
        for (const key of [...this.cache.keys()]) {
            if (key.includes(baseFilename) && 
                (key.startsWith('data_') || key.startsWith('daily_'))) {
                logger.debug(`Invalidating cache: ${key}`);
                this.cache.delete(key);
                this.cacheExpiry.delete(key);
            }
        }
    }

    getVersionFromUrl(url) {
        if (!url) return null;
        
        const match = url.match(/(\d{8})\.csv/);
        return match ? match[1] : null;
    }

    async createChartsProgressively(configArray) {
        try {
            logger.log('Creating charts progressively...');
            
            // Add loading indicators for all charts
            this.addLoadingIndicators(configArray);
            
            // Group charts by priority
            const highPriorityCharts = configArray.filter(c => 
                ['era5_t2m_anom_month_day'].includes(c.canvasId));
            const normalPriorityCharts = configArray.filter(c => 
                !highPriorityCharts.some(hpc => hpc.canvasId === c.canvasId));
            
            // CHANGE: First load only the high priority chart data
            logger.log('Loading high priority chart data first...');
            const priorityStart = performance.now();
            
            // Just fetch data for high priority charts
            for (const config of highPriorityCharts) {
                // Get URL and fetch data specifically for this chart
                const url = await this.getPresignedUrl(config.filename);
                
                if (config.type === 'daily_line') {
                    await this.fetchDailyData(config.filename, true); // Skip version check
                } else {
                    await this.fetchData(config.filename, config.x_var, config.y_var, true); // Skip version check
                }
            }
            
            logger.log(`Priority data loaded in ${(performance.now() - priorityStart).toFixed(0)}ms`);
            
            // Initialize high priority charts immediately
            logger.log('Creating high priority charts...');
            await Promise.all(
                highPriorityCharts.map(config => this.initializeChart(config))
            );
            logger.log('High priority charts loaded');
            
            // Update display for high priority charts
            this.updateDisplayValues(['era5_t2m_anom_month_day']);
            
            // CHANGE: Now in the background, preload remaining chart data
            logger.log('Preloading remaining chart data...');
            const remainingDataPromise = this.preloadData(normalPriorityCharts);
            
            // Start creating remaining charts, waiting for their data
            logger.log('Loading remaining charts...');
            
            // Use setTimeout to give the browser time to render the priority chart
            setTimeout(async () => {
                try {
                    // Make sure remaining data is loaded
                    await remainingDataPromise;
                    
                    // Now create all remaining charts
                    await Promise.all(
                        normalPriorityCharts.map(config => this.initializeChart(config))
                    );
                    logger.log('All charts loaded successfully');
                    
                    // Update display values for all charts
                    this.updateDisplayValues();
                } catch (error) {
                    logger.error('Error loading normal priority charts:', error);
                } finally {
                    // Always clean up loading indicators
                    document.querySelectorAll('.chart-loading').forEach(el => el.remove());
                }
            }, 100);
            
            // Start version check in parallel for remaining charts
            const versionCheckPromise = this.checkForNewVersions();
            
        } catch (error) {
            logger.error('Error creating charts:', error);
            document.querySelectorAll('.chart-loading').forEach(el => el.remove());
        }
    }

    // New helper method to add loading indicators
    addLoadingIndicators(configArray) {
        configArray.forEach(config => {
            const canvas = document.getElementById(config.canvasId);
            if (canvas) {
                const container = canvas.parentElement;
                if (container) {
                    const existingIndicator = container.querySelector('.chart-loading');
                    if (!existingIndicator) {
                        const loadingIndicator = document.createElement('div');
                        loadingIndicator.className = 'chart-loading';
                        loadingIndicator.textContent = 'Pulling latest data...';
                        loadingIndicator.style.cssText = 
                            'position: absolute; top: 50%; left: 50%; ' +
                            'transform: translate(-50%, -50%); color: #929190; ' + 
                            'font-family: "Space Mono", monospace;';
                        container.style.position = 'relative';
                        container.appendChild(loadingIndicator);
                    }
                }
            }
        });
    }

    // New helper to preload all chart data
    async preloadAllChartData(configArray) {
        // First get all URLs in parallel
        const urlPromises = configArray.map(config => 
            this.getPresignedUrl(config.filename)
        );
        await Promise.all(urlPromises);
        
        // Then preload all data in parallel
        return this.preloadData(configArray);
    }

    // Add a method to update display values inside the DeepSkyCharts class
    updateDisplayValues(metricIds = null) {
        // If no specific metrics provided, update all
        const metrics = metricIds || [
            'era5_t2m_anom_year',
            'aviso_slr',
            'arctic_sea_ice',
            'co2_ppm',
            'home_insurance_premium',
            'noaa_billion',
            'era5_sst_anom_month_day',
            'era5_t2m_anom_month_day'
        ];

        metrics.forEach(metricId => {
            // Update latest value display
            const latestData = this.getLatestValue(metricId);
            if (latestData) {
                const displayElement = document.getElementById(`${metricId}_latest_value`);
                if (displayElement) {
                    displayElement.textContent = latestData.formattedValue;
                    logger.log(`Updated ${metricId} value to ${latestData.formattedValue}`);
                }
            }

            // Update "last update" text
            const updateElement = document.getElementById(`${metricId}_last_update`);
            if (updateElement) {
                const lastUpdate = this.getLastUpdate(metricId);
                if (lastUpdate) {
                    updateElement.textContent = lastUpdate;
                }
            }
        });
    }
}

// Helper function to detect French language based on URL
function isLanguageFrench() {
    const isFrench = window.location.href.includes('://fr.deepskyclimate.com');
    return isFrench;
}

// Update ticker styles to be more flexible
const tickerStyles = `
    .news-ticker-container {
        width: 100%;
        overflow: hidden;
        min-height: 40px;
        height: auto;
        display: flex;
        align-items: center;
        position: relative;
        background-color: #151515;
        margin: 0;
        padding: 0;
    }

    .ticker-content {
        white-space: nowrap;
        overflow: hidden;
        width: 100%;
        padding: 0;
        margin: 0;
    }

    .news-item {
        display: inline-block;
        margin-right: 40px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1em;
        color: rgba(255, 255, 255, 0.7);
        text-transform: none;
        line-height: 1.4;
    }

    @keyframes ticker {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
`;

// Apply styles once
const styleSheet = document.createElement("style");
styleSheet.textContent = tickerStyles;
document.head.appendChild(styleSheet);