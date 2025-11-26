# Frontend Architecture Plan

## Overview

We are refactoring the legacy `chart_templates.js` file into a modular, maintainable, and performant frontend architecture. The goal is to reproduce all existing functionality while improving:

- **Modularity**: Separate concerns into focused modules
- **Debuggability**: Clear code organization and logging
- **DRY Principles**: Eliminate code duplication
- **Performance**: Reduce page loading time through optimization
- **Maintainability**: Easier to update and extend

## Current State Analysis

### Legacy System (`chart_templates.js`)
- **Single file**: ~1300 lines containing all functionality
- **Mixed concerns**: Data fetching, chart creation, styling, news ticker all in one place
- **Working features**: All charts render, news ticker scrolls, data displays correctly
- **Performance**: Sequential loading, slower initial page load

### Target Modular System
- **Bundle approach**: `chart_templates_modular_bundle.js` for webflow compatibility
- **Source modules**: Organized functionality in `webflow/src/` directory
- **Improved performance**: Parallel data loading, optimized caching

## Functionality Breakdown

### 1. Core Configuration
**Purpose**: Centralized constants, colors, fonts, formatters

**Current State**:
- **Legacy**: Defined at top of `chart_templates.js`
- **Modular Source**: `webflow/src/core/config.js`
- **Bundle**: Duplicated in `chart_templates_modular_bundle.js`

**Action Needed**: 
- Use single source of truth from modular config
- Remove duplication in bundle

### 2. Data Management
**Purpose**: Fetch, cache, and process chart data from S3/local sources

**Current State**:
- **Legacy**: `DeepSkyCharts.fetchData()`, `DeepSkyCharts.fetchDailyData()`
- **Modular Source**: `webflow/src/data/dataManager.js` - `DataManager` class
- **Bundle**: `DataManagerBundle` class (duplicate implementation)

**Functions**:
- Get presigned URLs for CSV files
- Fetch and parse CSV data
- Cache data with expiry
- Process daily vs regular chart data
- Version checking for updates

**Action Needed**:
- Eliminate `DataManagerBundle` duplication
- Use `webflow/src/data/dataManager.js` as single source

### 3. Chart Templates & Styling
**Purpose**: Define chart configurations, axis settings, tooltips

**Current State**:
- **Legacy**: `chartTemplates` object and helper functions in `chart_templates.js`
- **Modular Source**: `webflow/src/charts/chartTemplates.js`
- **Bundle**: Missing proper chart templates (using basic fallback)

**Functions**:
- Chart type definitions (bar, line, daily_line)
- Axis configuration
- Tooltip formatting
- Chart.js options

**Action Needed**:
- Import complete chart templates from modular source
- Remove basic fallback implementations

### 4. Chart Creation & Management
**Purpose**: Initialize Chart.js instances, manage chart lifecycle

**Current State**:
- **Legacy**: `DeepSkyCharts.initializeChart()`, `DeepSkyCharts.buildChartConfig()`
- **Modular Source**: `webflow/src/deepSkyCharts.js` - `DeepSkyCharts` class
- **Bundle**: `DeepSkyChartsOptimized` class (partial duplication)

**Functions**:
- Create individual charts
- Build chart configurations
- Manage chart instances
- Update display values
- Progressive loading with optimization

**Action Needed**:
- Reduce duplication between optimized and modular versions
- Clear delegation strategy

### 5. News Ticker
**Purpose**: Fetch and display scrolling climate news headlines

**Current State**:
- **Legacy**: `NewsTicker` class in `chart_templates.js`
- **Modular Source**: `webflow/src/components/newsTicker.js`
- **Bundle**: Complete `NewsTicker` implementation (duplicate)

**Functions**:
- Fetch RSS feed via API
- Fallback to local stories for development
- Horizontal scrolling animation
- Format: "Source • Date | Title"

**Action Needed**:
- Remove duplicate implementation from bundle
- Reference modular source

### 6. Logging & Utilities
**Purpose**: Consistent logging, error handling, utilities

**Current State**:
- **Legacy**: Basic console logging throughout
- **Modular Source**: `webflow/src/core/logger.js`
- **Bundle**: Simple logger object

**Action Needed**:
- Use modular logger consistently
- Remove basic implementation

## Proposed Architecture

### Bundle Structure (`chart_templates_modular_bundle.js`)
The bundle should be a **thin orchestration layer** that:

1. **Imports/Includes** all modular functionality in correct order
2. **Coordinates** the initialization process
3. **Optimizes** loading with parallel data fetching
4. **Exports** main classes for webflow compatibility

### Module Organization (`webflow/src/`)
```
webflow/src/
├── core/
│   ├── config.js          # Constants, colors, formatters
│   └── logger.js          # Logging utilities
├── components/
│   └── newsTicker.js      # News ticker component
├── charts/
│   └── chartTemplates.js  # Chart type definitions
├── data/
│   └── dataManager.js     # Data fetching and caching
├── styles/
│   └── tickerStyles.js    # CSS style injection
└── deepSkyCharts.js       # Main chart management class
```

## Implementation Strategy

### Phase 1: Audit Current Bundle
- [ ] Identify all duplicated functionality
- [ ] Map each function to its modular source
- [ ] Document what's missing from modular sources

### Phase 2: Eliminate Duplication
- [ ] Remove duplicate implementations from bundle
- [ ] Include modular code in correct dependency order
- [ ] Maintain webflow compatibility (no ES6 imports)

### Phase 3: Optimization Layer
- [ ] Keep performance optimizations (parallel loading)
- [ ] Maintain loading indicators
- [ ] Preserve fast value display

### Phase 4: Testing & Validation
- [ ] Verify all charts render correctly
- [ ] Confirm news ticker scrolls properly
- [ ] Test loading performance improvements
- [ ] Validate local development workflow

## Success Criteria

1. **Functionality**: All features from legacy system work correctly and the page matches the aesthetics of the existing climate dashboard page
2. **Performance**: Page loads faster than legacy version
3. **Modularity**: Clear separation of concerns
4. **Maintainability**: Easy to modify individual components
5. **DRY**: No duplicated functionality
6. **Debuggability**: Clear error messages and logging

## File Dependencies

### Current Bundle Dependencies
- `chart_templates_modular_bundle.js` should depend on:
  - All `webflow/src/` modules (embedded for webflow compatibility)
  - Chart.js library (external)
  - DOM elements (HTML structure)

### Deployment Strategy
- Bundle includes all modular code (no runtime imports)
- Single file for webflow compatibility
- Optimized for production performance
- Clear development workflow with modular sources

## Next Steps

1. **Complete this documentation** with specific function mappings
2. **Audit current bundle** for exact duplication points
3. **Create clean bundle** that properly orchestrates modular components
4. **Test thoroughly** to ensure no regressions
5. **Deploy to test environment** for validation

This architecture will give us the benefits of modular development while maintaining webflow compatibility and improving performance.