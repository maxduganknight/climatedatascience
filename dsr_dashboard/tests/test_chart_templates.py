import pytest
from pathlib import Path
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import hashlib
from PIL import Image
import numpy as np
import time

class TestChartTemplates:
    @pytest.fixture(scope="class")
    def chart_templates_js(self):
        """Load the chart_templates.js file content"""

        current_dir = Path(__file__).resolve().parent        
        js_path = current_dir.parent / "webflow" / "chart_templates.js"
        with open(js_path, "r") as f:
            return f.read()
    
    @pytest.fixture(scope="class")
    def browser(self):
        """Setup headless Chrome browser for testing"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    @pytest.fixture(scope="class")
    def test_page(self, tmp_path_factory, chart_templates_js):
        """Create a test HTML page with charts"""
        tmp_dir = tmp_path_factory.mktemp("web")
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chart Templates Test</title>
            <meta charset="UTF-8">
            <link href="https://fonts.googleapis.com/css2?family=Space+Mono&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
            <style>
                * {
                    font-family: 'Space Mono', monospace;
                    text-transform: uppercase;
                }
                body {
                    margin: 0;
                    padding: 20px;
                    background-color: #111111;
                }
                .chart-container {
                    position: relative;
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                    margin-bottom: 40px;
                    height: 500px !important;
                    width: 100%;
                }
                canvas {
                    height: 100% !important;
                    width: 100% !important;
                }
            </style>
        </head>
        <body>
            <div class="chart-container">
                <canvas id="test-bar-chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="test-line-chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="test-daily-line-chart"></canvas>
            </div>
            <div id="newsTicker"></div>

            <script>
                // First, load chart templates
                {chart_templates_js}

                // Initialize test data
                const testData = {
                    bar: {
                        labels: ['2020', '2021', '2022', '2023'],
                        datasets: [{
                            label: 'Test Bar Chart',
                            data: [10, 15, 20, 25],
                            backgroundColor: '#EC6252'
                        }]
                    },
                    line: {
                        labels: ['Jan', 'Feb', 'Mar', 'Apr'],
                        datasets: [{
                            label: 'Test Line Chart',
                            data: [1, 2, 3, 4],
                            borderColor: '#EC6252',
                            tension: 0.3
                        }]
                    },
                    daily_line: {
                        datasets: [
                            {
                                label: '2023',
                                data: [
                                    {x: '01-01', y: 1},
                                    {x: '02-01', y: 2},
                                    {x: '03-01', y: 3}
                                ],
                                borderColor: '#505151'
                            },
                            {
                                label: '2024',
                                data: [
                                    {x: '01-01', y: 2},
                                    {x: '02-01', y: 3},
                                    {x: '03-01', y: 4}
                                ],
                                borderColor: '#EC6252'
                            }
                        ]
                    }
                };

                // Initialize charts
                window.onload = function() {
                    // Bar chart
                    new Chart(
                        document.getElementById('test-bar-chart'),
                        {
                            type: 'bar',
                            data: testData.bar,
                            options: chartTemplates.bar.defaultOptions('units')
                        }
                    );

                    // Line chart
                    new Chart(
                        document.getElementById('test-line-chart'),
                        {
                            type: 'line',
                            data: testData.line,
                            options: chartTemplates.line.defaultOptions('units')
                        }
                    );

                    // Daily line chart
                    new Chart(
                        document.getElementById('test-daily-line-chart'),
                        {
                            type: 'line',
                            data: testData.daily_line,
                            options: chartTemplates.daily_line.defaultOptions('units')
                        }
                    );
                };
            </script>
        </body>
        </html>
        """
        
        html_path = tmp_dir / "test.html"
        
        # Write the HTML file with the chart_templates_js content inserted
        with open(html_path, "w") as f:
            f.write(html_content.replace("{chart_templates_js}", chart_templates_js))
        
        return html_path

    def test_chart_types_exist(self, chart_templates_js):
        """Test that all chart types are defined"""
        required_types = ['bar', 'line', 'daily_line']
        # Look for the chartTemplates object definition
        assert 'const chartTemplates = {' in chart_templates_js, "chartTemplates object not found"
        
        for chart_type in required_types:
            # Check if each chart type is defined as a property in chartTemplates
            assert f"    {chart_type}: {{" in chart_templates_js, f"Chart type '{chart_type}' not found in chartTemplates"
            # Check if each chart type has required properties
            assert f"        type: '{chart_type if chart_type != 'daily_line' else 'line'}'" in chart_templates_js, \
                f"Chart type '{chart_type}' missing type property"
            assert "        defaultOptions: " in chart_templates_js, \
                f"Chart type '{chart_type}' missing defaultOptions"

    # def test_visual_regression(self, browser, test_page):
    #     """Test visual appearance of charts hasn't changed"""
    #     browser.get(f"file://{test_page}")
        
    #     # Set a specific window size for consistent screenshots
    #     browser.set_window_size(1200, 2000)
        
    #     # Wait longer for charts to render and animations to complete
    #     WebDriverWait(browser, 10).until(
    #         EC.presence_of_element_located((By.ID, "test-bar-chart"))
    #     )
        
    #     # Wait for fonts to load and animations to complete
    #     time.sleep(3)  # Increased wait time
        
    #     # Execute JavaScript to ensure all Chart.js animations are complete
    #     browser.execute_script("""
    #         document.fonts.ready.then(function() {
    #             console.log('Fonts loaded');
    #         });
    #     """)
        
    #     # Take screenshot
    #     screenshot = browser.get_screenshot_as_png()
        
    #     # Calculate hash of screenshot
    #     screenshot_hash = hashlib.md5(screenshot).hexdigest()
        
    #     # Save both hash and image
    #     baseline_dir = Path("tests/baselines")
    #     baseline_dir.mkdir(parents=True, exist_ok=True)
        
    #     hash_path = baseline_dir / "chart_templates_baseline.md5"
    #     image_path = baseline_dir / "chart_templates_baseline.png"
    #     current_image_path = baseline_dir / "chart_templates_current.png"
        
    #     print(f"Writing current screenshot to: {current_image_path}")
    #     # Always save current screenshot
    #     with open(current_image_path, "wb") as f:
    #         f.write(screenshot)
    #     print(f"Current screenshot saved: {current_image_path.exists()}")
        
    #     # First run or force update baseline
    #     if not hash_path.exists() or os.getenv('UPDATE_BASELINE'):
    #         print(f"Writing baseline files to: {hash_path} and {image_path}")
    #         with open(hash_path, "w") as f:
    #             f.write(screenshot_hash)
    #         with open(image_path, "wb") as f:
    #             f.write(screenshot)
    #         print(f"Baseline files saved: {hash_path.exists()} {image_path.exists()}")
    #         pytest.skip("Baseline created/updated")
        
    #     with open(hash_path, "r") as f:
    #         baseline_hash = f.read().strip()
        
    #     assert screenshot_hash == baseline_hash, (
    #         "Visual appearance of charts has changed. "
    #         f"Compare images:\n"
    #         f"Baseline: {image_path}\n"
    #         f"Current:  {current_image_path}\n"
    #         "To update baseline, run with UPDATE_BASELINE=1"
    #     )

    def test_news_ticker_configuration(self, chart_templates_js):
        """Test news ticker styling and configuration"""
        assert "class NewsTicker" in chart_templates_js
        
        assert "itemCount: 10" in chart_templates_js
        
        # Test styling
        assert "font-family: 'IBM Plex Sans', sans-serif" in chart_templates_js
        assert "animation: ticker" in chart_templates_js

    def test_chart_formatters(self, chart_templates_js):
        """Test value formatters configuration"""
        formatters = {
            'co2_ppm': 'Math.round',
            'aviso_slr': 'Math.round',
            'era5_t2m_anom_year': 'toFixed(2)',
            'era5_sst_anom_month_day': 'toFixed(2)'
        }
        
        for metric, formatter in formatters.items():
            assert f"'{metric}'" in chart_templates_js
            assert formatter in chart_templates_js 