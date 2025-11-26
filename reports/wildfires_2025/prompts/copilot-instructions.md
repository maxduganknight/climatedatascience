# Copilot Instructions for Wildfires 2025 Project

## Code Structure Guidelines

### Modularity
- Extract reusable code into functions
- Place shared functions in utils.py and import as needed
- Each function should have a single responsibility
- Use appropriate function parameters instead of relying on global state

### Simplicity
- Favor straightforward implementations over complex ones
- Apply the principle of least astonishment
- Use clear, descriptive variable and function names
- Keep functions small and focused (ideally under 30 lines)

### Standard File Structure
```python
# Import statements
import pandas as pd
import numpy as np
from scripts.utils import setup_enhanced_plot  # project-specific imports

# Global variables (constants)
RISK_CATEGORIES = ['Extreme Fire Risk', 'High Fire Risk', 'Low Fire Risk']
DATA_PATH = 'data/processed/risk_data.csv'

# Helper functions
def process_data(data):
    """Process the input data."""
    # Implementation

def analyze_results(processed_data):
    """Analyze the processed data."""
    # Implementation

# Main function
def main():
    """Main execution function."""
    data = pd.read_csv(DATA_PATH)
    processed_data = process_data(data)
    results = analyze_results(processed_data)
    return results

# Script execution
if __name__ == "__main__":
    main()
```

### Code Debugging
- Include informative docstrings for all functions
- Use meaningful error messages
- Add logging at appropriate levels
- Include type hints where helpful

### Changes to Existing Code
- Prefer minimal changes to existing code
- Maintain backward compatibility when possible
- Document reasons for significant changes

## Best Practices
- Follow PEP 8 style guidelines
- Use consistent naming conventions (snake_case for functions/variables, all caps for global variables)
- Add comments for complex logic, not for obvious operations
- Ensure proper exception handling

## Project-Specific Guidelines
- Use the provided functions in utils.py for consistent 
processing and visualization across different datasets and scripts
