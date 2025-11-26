#!/usr/bin/env python3
"""
Heatwave Mapper - Scrapes heat wave warning data from NWS and creates map visualizations
Combines NWS heat wave weather alerts with geographic data for mapping heat wave risk.

Author: Deep Sky Research
Date: 2025-06-23
"""

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import argparse
import sys
import os
from matplotlib.patches import Patch
from datetime import datetime
import pytz

# Add path for importing from other Deep Sky projects
sys.path.append('/Users/max/Deep_Sky/')
sys.path.append('/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/wildfires_2025/scripts/')

# Import Deep Sky utilities and branding
from utils import (
    setup_space_mono_font, setup_enhanced_plot, format_plot_title, 
    add_deep_sky_branding, save_plot, COLORS, RISK_COLORS
)

# Import for fuzzy string matching
from difflib import SequenceMatcher


class CountyMatcher:
    """
    Maps heat wave alerts to county FIPS codes using NWS geocoding data and county name matching.
    
    Uses a two-tier approach:
    1. Primary: Extract SAME codes from NWS geocode field (most reliable)
    2. Fallback: Parse area descriptions and match county names
    
    Special handling for Connecticut counties vs planning regions.
    """
    
    def __init__(self, county_csv_path: str):
        """
        Initialize the county matcher with county reference data.
        
        Args:
            county_csv_path (str): Path to CSV with county FIPS codes and names
        """
        self.county_csv_path = county_csv_path
        self.county_df = None
        self.county_lookup = {}
        self.state_lookup = {}
        self.ct_county_mapping = self._create_ct_county_mapping()
        self._load_county_data()
    
    def _create_ct_county_mapping(self):
        """
        Create mapping of old Connecticut county FIPS codes to new planning region FIPS codes.
        Based on research from ctdata.org about Connecticut's transition from counties to planning regions.
        
        Returns:
            Dict: Mapping from old county FIPS to new planning region FIPS
        """
        # Mapping based on geographic overlap and historical data
        # Old FIPS -> New Planning Region FIPS
        ct_mapping = {
            '09001': '09190',  # Fairfield County -> Western Connecticut Planning Region
            '09003': '09120',  # Hartford County -> Greater Bridgeport Planning Region (partial) + Capitol Planning Region (partial)
            '09005': '09180',  # Litchfield County -> Southeastern Connecticut Planning Region (partial) + Northwest Hills Planning Region (partial)
            '09007': '09110',  # Middlesex County -> Capitol Planning Region (partial) + Lower Connecticut River Valley Planning Region (partial)
            '09009': '09120',  # New Haven County -> Greater Bridgeport Planning Region + South Central Connecticut Planning Region
            '09011': '09180',  # New London County -> Southeastern Connecticut Planning Region
            '09013': '09150',  # Tolland County -> Northeastern Connecticut Planning Region
            '09015': '09140'   # Windham County -> Naugatuck Valley Planning Region (partial) + Northeastern Connecticut Planning Region (partial)
        }
        
        # For alerts, we'll create entries for both old and new FIPS codes
        # This allows us to maintain backward compatibility while supporting current data
        return ct_mapping
    
    def _load_county_data(self):
        """Load and prepare county reference data for matching."""
        try:
            self.county_df = pd.read_csv(self.county_csv_path)
            print(f"Loaded {len(self.county_df)} county records from {self.county_csv_path}")
            
            # Create lookup dictionaries for faster matching
            # Remove "County" suffix and standardize names for matching
            for _, row in self.county_df.iterrows():
                fips = str(row['combined_fips']).zfill(5)  # Ensure 5-digit format
                county_name = row['county_name'].replace(' County', '').strip()
                state_name = row['state_name'].strip()
                
                # Create county name lookup by state
                if state_name not in self.county_lookup:
                    self.county_lookup[state_name] = {}
                
                self.county_lookup[state_name][county_name.lower()] = fips
                
                # Create reverse lookup for state by county name (for disambiguation)
                county_key = county_name.lower()
                if county_key not in self.state_lookup:
                    self.state_lookup[county_key] = []
                self.state_lookup[county_key].append(state_name)
            
            # Add old Connecticut counties to the lookup for backward compatibility
            self._add_old_connecticut_counties()
                
        except Exception as e:
            print(f"Error loading county data: {e}")
            self.county_df = pd.DataFrame()
    
    def _add_old_connecticut_counties(self):
        """
        Add old Connecticut county names and FIPS codes to the lookup for NWS alert compatibility.
        This ensures alerts using old Connecticut county SAME codes are properly mapped.
        """
        # Old Connecticut counties with their FIPS codes and names
        old_ct_counties = {
            '09001': 'Fairfield',
            '09003': 'Hartford', 
            '09005': 'Litchfield',
            '09007': 'Middlesex',
            '09009': 'New Haven',
            '09011': 'New London',
            '09013': 'Tolland',
            '09015': 'Windham'
        }
        
        # Add these to our county DataFrame and lookup tables
        if 'Connecticut' not in self.county_lookup:
            self.county_lookup['Connecticut'] = {}
        
        for old_fips, old_name in old_ct_counties.items():
            # Add to lookup table
            self.county_lookup['Connecticut'][old_name.lower()] = old_fips
            
            # Add to state lookup for reverse matching
            county_key = old_name.lower()
            if county_key not in self.state_lookup:
                self.state_lookup[county_key] = []
            if 'Connecticut' not in self.state_lookup[county_key]:
                self.state_lookup[county_key].append('Connecticut')
            
            # Create DataFrame row for the old county
            old_county_row = pd.DataFrame({
                'combined_fips': [old_fips],
                'county_name': [f'{old_name} County'],
                'state_name': ['Connecticut'],
                'area_sq_miles': [0]  # Placeholder value
            })
            
            # Add to the main county DataFrame
            self.county_df = pd.concat([self.county_df, old_county_row], ignore_index=True)
        
        print(f"Added {len(old_ct_counties)} old Connecticut counties for NWS alert compatibility")
    
    def same_codes_to_fips(self, same_codes: List[str]) -> List[str]:
        """
        Convert SAME codes to 5-digit county FIPS codes.
        
        Args:
            same_codes (List[str]): List of 6-digit SAME codes from NWS
            
        Returns:
            List[str]: List of 5-digit county FIPS codes
        """
        fips_codes = []
        for same_code in same_codes:
            if len(same_code) == 6:
                # Convert 6-digit SAME code to 5-digit FIPS
                # Remove leading zero from state code if present
                state_fips = same_code[:3].lstrip('0').zfill(2)
                county_fips = same_code[3:]
                fips_code = state_fips + county_fips
                fips_codes.append(fips_code)
            else:
                print(f"Invalid SAME code format: {same_code}")
        
        return fips_codes
    
    def extract_state_from_sender(self, sender_name: str) -> Optional[str]:
        """
        Extract state information from NWS sender name.
        
        Args:
            sender_name (str): NWS office name (e.g., "NWS Morristown TN")
            
        Returns:
            Optional[str]: State abbreviation or None
        """
        if not sender_name:
            return None
        
        # Common patterns: "NWS [City] [STATE]" or "NWS [City/State]"
        parts = sender_name.split()
        if len(parts) >= 2:
            # Look for 2-letter state abbreviation
            for part in parts:
                if len(part) == 2 and part.isupper():
                    # Convert state abbreviation to full name
                    state_abbrev_to_name = {
                        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
                        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
                        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
                        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
                        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
                        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
                        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
                        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
                        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
                        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
                        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
                        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
                        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
                    }
                    return state_abbrev_to_name.get(part)
        
        return None
    
    def match_county_names(self, area_desc: str, state_hint: Optional[str] = None) -> List[str]:
        """
        Match county names from area description to FIPS codes.
        
        Args:
            area_desc (str): Semicolon-separated area description from NWS
            state_hint (Optional[str]): State name hint for disambiguation
            
        Returns:
            List[str]: List of matched county FIPS codes
        """
        if not area_desc or self.county_df.empty:
            return []
        
        # Parse county names from area description
        county_names = [name.strip() for name in area_desc.split(';')]
        matched_fips = []
        
        for county_name in county_names:
            # Clean up county name - remove directional prefixes and other modifiers
            clean_name = self._clean_county_name(county_name)
            
            # Try exact match first
            fips_code = self._find_exact_match(clean_name, state_hint)
            if fips_code:
                matched_fips.append(fips_code)
                continue
            
            # Try fuzzy matching if exact match fails
            fips_code = self._find_fuzzy_match(clean_name, state_hint)
            if fips_code:
                matched_fips.append(fips_code)
            else:
                print(f"Could not match county: '{county_name}' (cleaned: '{clean_name}')")
        
        return matched_fips
    
    def _clean_county_name(self, county_name: str) -> str:
        """Clean and standardize county name for matching."""
        # Remove common directional prefixes
        prefixes_to_remove = [
            'Northern', 'Southern', 'Eastern', 'Western', 'Central',
            'North', 'South', 'East', 'West', 'Northwest', 'Northeast',
            'Southwest', 'Southeast', 'Upper', 'Lower', 'Inland'
        ]
        
        name = county_name.strip()
        for prefix in prefixes_to_remove:
            if name.startswith(prefix + ' '):
                name = name[len(prefix):].strip()
                break
        
        return name
    
    def _find_exact_match(self, county_name: str, state_hint: Optional[str] = None) -> Optional[str]:
        """Find exact county name match."""
        county_lower = county_name.lower()
        
        # If we have a state hint, search within that state first
        if state_hint and state_hint in self.county_lookup:
            if county_lower in self.county_lookup[state_hint]:
                return self.county_lookup[state_hint][county_lower]
        
        # Search across all states
        for state_name, counties in self.county_lookup.items():
            if county_lower in counties:
                return counties[county_lower]
        
        return None
    
    def _find_fuzzy_match(self, county_name: str, state_hint: Optional[str] = None, threshold: float = 0.8) -> Optional[str]:
        """Find fuzzy county name match using string similarity."""
        county_lower = county_name.lower()
        best_match = None
        best_score = 0
        
        # Search in hinted state first
        if state_hint and state_hint in self.county_lookup:
            for county, fips in self.county_lookup[state_hint].items():
                score = SequenceMatcher(None, county_lower, county).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = fips
        
        # If no good match in hinted state, search all states
        if not best_match:
            for state_name, counties in self.county_lookup.items():
                for county, fips in counties.items():
                    score = SequenceMatcher(None, county_lower, county).ratio()
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = fips
        
        return best_match
    
    def get_county_fips_for_alert(self, alert: Dict) -> List[str]:
        """
        Get county FIPS codes for a heat alert using all available methods.
        
        Args:
            alert (Dict): Heat alert data from NWS API
            
        Returns:
            List[str]: List of county FIPS codes
        """
        properties = alert.get('properties', {})
        
        # Method 1: Try SAME codes from geocode field (most reliable)
        geocode = properties.get('geocode', {})
        same_codes = geocode.get('SAME', [])
        
        if same_codes:
            fips_codes = self.same_codes_to_fips(same_codes)
            if fips_codes:
                return fips_codes
        
        # Method 2: Fallback to area description matching
        area_desc = properties.get('areaDesc', '')
        sender_name = properties.get('senderName', '')
        
        # Extract state hint from sender
        state_hint = self.extract_state_from_sender(sender_name)
        
        # Match county names
        fips_codes = self.match_county_names(area_desc, state_hint)
        
        return fips_codes


class HeatWaveDataRetriever:
    """
    Retrieves and processes heat wave alert data from the National Weather Service API.
    
    This class extends the functionality from weather_alert_tracker.py to focus specifically
    on heat-related weather alerts and their geographic boundaries.
    """
    
    def __init__(self):
        """Initialize the heat wave data retriever."""
        self.nws_alerts_url = "https://api.weather.gov/alerts/active"
        self.processed_alerts = set()  # Track processed alerts to avoid duplicates
        
        # Define heat-related weather events to monitor
        self.heat_events = {
            "Excessive Heat Warning",
            "Excessive Heat Watch", 
            "Heat Advisory",
            "Extreme Heat Warning",
            "Extreme Heat Watch"
        }
        
        # Define heat risk categories and their colors
        self.heat_risk_colors = {
            'Extreme Heat Risk': '#8B0000',     # Dark red
            'High Heat Risk': COLORS['primary'], # Red
            'Moderate Heat Risk': COLORS['tertiary'], # Yellow/gold
            'Low Heat Risk': COLORS['secondary']  # Green
        }
    
    def fetch_alerts(self) -> List[Dict]:
        """
        Fetch current alerts from NWS API.
        
        Returns:
            List[Dict]: List of alert features from the NWS API
        """
        try:
            response = requests.get(
                self.nws_alerts_url,
                headers={"User-Agent": "HeatWaveMapper/1.0 (Deep Sky Research)"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            alerts = data.get("features", [])
            print(f"Fetched {len(alerts)} total alerts from NWS API")
            return alerts
            
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []
    
    def is_heat_alert(self, alert: Dict) -> bool:
        """
        Check if alert is heat-related.
        
        Args:
            alert (Dict): Alert feature from NWS API
            
        Returns:
            bool: True if alert is heat-related
        """
        properties = alert.get("properties", {})
        event_type = properties.get("event", "")
        severity = properties.get("severity", "")
        
        is_heat_event = event_type in self.heat_events
        is_significant = severity in ["Extreme", "Severe", "Moderate"]
        
        return is_heat_event and is_significant
    
    def extract_alert_geometry(self, alert: Dict) -> Optional[Dict]:
        """
        Extract geographic information from alert.
        
        Args:
            alert (Dict): Alert feature from NWS API
            
        Returns:
            Optional[Dict]: Geographic information or None if not available
        """
        geometry = alert.get("geometry")
        properties = alert.get("properties", {})
        
        # If no geometry, try to get from properties
        if not geometry:
            # Some alerts might have geographical info in properties
            print(f"No geometry found for alert: {properties.get('event', 'Unknown')}")
            # For now, we'll skip alerts without geometry, but we could extend this
            # to use areaDesc or other location information
            return None
            
        return {
            'geometry': geometry,
            'event_type': properties.get("event", ""),
            'severity': properties.get("severity", ""),
            'areas': properties.get("areaDesc", ""),
            'effective': properties.get("effective", ""),
            'expires': properties.get("expires", ""),
            'headline': properties.get("headline", ""),
            'description': properties.get("description", "")
        }
    
    def categorize_heat_risk(self, alert_data: Dict) -> str:
        """
        Categorize heat risk level based on alert type and severity.
        
        Args:
            alert_data (Dict): Alert data with event type and severity
            
        Returns:
            str: Heat risk category
        """
        event_type = alert_data.get('event_type', '')
        severity = alert_data.get('severity', '')
        
        # Extreme risk for extreme warnings
        if 'Extreme' in event_type or severity == 'Extreme':
            return 'Extreme Heat Risk'
        
        # High risk for excessive heat warnings
        if 'Excessive' in event_type and 'Warning' in event_type:
            return 'High Heat Risk'
        
        # Moderate risk for advisories and watches
        if 'Advisory' in event_type or 'Watch' in event_type:
            return 'Moderate Heat Risk'
        
        # Default to low risk
        return 'Low Heat Risk'
    
    def debug_alert_types(self) -> Dict[str, int]:
        """
        Debug function to see what types of alerts are currently available.
        
        Returns:
            Dict[str, int]: Count of each alert type
        """
        all_alerts = self.fetch_alerts()
        alert_types = {}
        heat_alert_details = []
        
        for alert in all_alerts:
            properties = alert.get("properties", {})
            event_type = properties.get("event", "Unknown") 
            alert_types[event_type] = alert_types.get(event_type, 0) + 1
            
            # Check heat alerts specifically
            if self.is_heat_alert(alert):
                geometry = alert.get("geometry")
                heat_alert_details.append({
                    'event': event_type,
                    'severity': properties.get("severity", ""),
                    'areas': properties.get("areaDesc", "")[:100] + "..." if len(properties.get("areaDesc", "")) > 100 else properties.get("areaDesc", ""),
                    'has_geometry': geometry is not None
                })
        
        print(f"Current alert types:")
        for event_type, count in sorted(alert_types.items()):
            print(f"  {event_type}: {count}")
        
        if heat_alert_details:
            print(f"\nHeat alert details:")
            for i, alert in enumerate(heat_alert_details[:10]):  # Show first 10
                print(f"  {i+1}. {alert['event']} ({alert['severity']}) - Has geometry: {alert['has_geometry']}")
                print(f"     Areas: {alert['areas']}")
        else:
            print(f"\nNo heat alerts found matching criteria")
        
        return alert_types
    
    def get_heat_alerts_with_geometry(self) -> List[Dict]:
        """
        Get all current heat alerts with their geographic boundaries.
        
        Returns:
            List[Dict]: List of heat alerts with geometry and risk categories
        """
        all_alerts = self.fetch_alerts()
        heat_alerts = []
        
        for alert in all_alerts:
            if self.is_heat_alert(alert):
                alert_data = self.extract_alert_geometry(alert)
                if alert_data:
                    # Add risk categorization
                    alert_data['risk_category'] = self.categorize_heat_risk(alert_data)
                    heat_alerts.append(alert_data)
        
        print(f"Found {len(heat_alerts)} heat-related alerts with geographic data")
        return heat_alerts


class HeatWaveMapper:
    """
    Creates map visualizations of heat wave risk areas.
    
    This class combines heat wave alert data with geographic boundaries to create
    maps showing heat wave risk by location using Deep Sky branding.
    """
    
    def __init__(self, data_retriever: HeatWaveDataRetriever, county_csv_path: str = None):
        """
        Initialize the heat wave mapper.
        
        Args:
            data_retriever (HeatWaveDataRetriever): Data retriever instance
            county_csv_path (str, optional): Path to county FIPS CSV file
        """
        self.data_retriever = data_retriever
        self.font_props = setup_space_mono_font()
        
        # Initialize county matcher if CSV path provided
        self.county_matcher = None
        if county_csv_path:
            self.county_matcher = CountyMatcher(county_csv_path)
    
    def format_timestamp(self, timestamp_str: str) -> str:
        """
        Convert ISO timestamp to readable Eastern Time format.
        
        Args:
            timestamp_str (str): ISO format timestamp (e.g., "2025-06-23T11:36:00-04:00")
            
        Returns:
            str: Formatted timestamp in Eastern Time (e.g., "Jun 23, 2025 11:36 AM EDT")
        """
        if not timestamp_str or pd.isna(timestamp_str):
            return ""
        
        try:
            # Parse ISO format timestamp (handles timezone info)
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Define Eastern timezone (handles EDT/EST automatically)
            eastern_tz = pytz.timezone('US/Eastern')
            
            # Convert to Eastern time
            dt_eastern = dt.astimezone(eastern_tz)
            
            # Format as readable string with Eastern timezone
            formatted = dt_eastern.strftime("%b %d, %Y %I:%M %p %Z")
            
            return formatted
            
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}")
            return timestamp_str  # Return original if parsing fails
    
    def get_alert_priority_score(self, event_type: str, severity: str) -> int:
        """
        Calculate priority score for alert ranking. Higher score = higher priority.
        
        Args:
            event_type (str): NWS event type (e.g., "Extreme Heat Warning")
            severity (str): NWS severity level (e.g., "Severe")
            
        Returns:
            int: Priority score for ranking alerts
        """
        # Event type priority (higher number = higher priority)
        event_priorities = {
            'Extreme Heat Warning': 30,
            'Extreme Heat Watch': 20,
            'Heat Advisory': 10
        }
        
        # Severity priority (higher number = higher priority)
        severity_priorities = {
            'Severe': 3,
            'Moderate': 2,
            'Minor': 1
        }
        
        event_score = event_priorities.get(event_type, 0)
        severity_score = severity_priorities.get(severity, 0)
        
        # Combined score: event priority * 10 + severity priority
        return event_score + severity_score
    
    def alerts_to_geodataframe(self, heat_alerts: List[Dict]) -> gpd.GeoDataFrame:
        """
        Convert heat alerts to a GeoDataFrame for mapping.
        
        Args:
            heat_alerts (List[Dict]): List of heat alerts with geometry
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with alert geometries and attributes
        """
        if not heat_alerts:
            print("No heat alerts to convert to GeoDataFrame")
            return gpd.GeoDataFrame()
        
        # Prepare data for GeoDataFrame
        rows = []
        for alert in heat_alerts:
            geometry = alert.get('geometry')
            if geometry:
                rows.append({
                    'event_type': alert.get('event_type', ''),
                    'severity': alert.get('severity', ''),
                    'areas': alert.get('areas', ''),
                    'risk_category': alert.get('risk_category', ''),
                    'effective': alert.get('effective', ''),
                    'expires': alert.get('expires', ''),
                    'headline': alert.get('headline', ''),
                    'geometry': geometry
                })
        
        if not rows:
            print("No valid geometries found in heat alerts")
            return gpd.GeoDataFrame()
        
        # Convert to GeoDataFrame
        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry'].astype(str)))
        
        # Set CRS to WGS84 (standard for NWS data)
        gdf.set_crs(epsg=4326, inplace=True)
        
        print(f"Created GeoDataFrame with {len(gdf)} heat alert polygons")
        return gdf
    
    def create_heat_risk_map(self, output_path: str, 
                           title: str = "CURRENT HEAT WAVE ALERTS",
                           subtitle: str = "NATIONAL WEATHER SERVICE HEAT WARNINGS AND ADVISORIES",
                           figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """
        Create a map visualization of current heat wave alerts.
        
        Args:
            output_path (str): Path to save the map image
            title (str): Map title
            subtitle (str): Map subtitle
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: The created figure
        """
        # Get heat alerts data
        heat_alerts = self.data_retriever.get_heat_alerts_with_geometry()
        
        if not heat_alerts:
            print("No heat alerts found. Creating empty map.")
            return self._create_empty_map(output_path, title, subtitle, figsize)
        
        # Convert to GeoDataFrame
        alerts_gdf = self.alerts_to_geodataframe(heat_alerts)
        
        if alerts_gdf.empty:
            print("No valid alert geometries found. Creating empty map.")
            return self._create_empty_map(output_path, title, subtitle, figsize)
        
        # Set up the figure with Deep Sky styling
        fig, ax, font_props = setup_enhanced_plot(figsize=figsize)
        
        # Load US states for context
        try:
            print("Loading US state boundaries for context...")
            states_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip')
            
            # Filter to continental US
            continental_states = states_gdf[~states_gdf['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
            
            # Convert to Web Mercator for better visualization
            continental_states = continental_states.to_crs(epsg=3857)
            alerts_gdf = alerts_gdf.to_crs(epsg=3857)
            
            # Plot state boundaries
            continental_states.boundary.plot(ax=ax, linewidth=0.5, color='#CCCCCC', alpha=0.8)
            
        except Exception as e:
            print(f"Could not load state boundaries: {e}")
            continental_states = None
        
        # Plot heat alerts by risk category
        risk_categories = alerts_gdf['risk_category'].unique()
        
        for category in risk_categories:
            category_data = alerts_gdf[alerts_gdf['risk_category'] == category]
            color = self.data_retriever.heat_risk_colors.get(category, COLORS['comparison'])
            
            category_data.plot(
                ax=ax,
                color=color,
                alpha=0.7,
                linewidth=0.5,
                edgecolor='white',
                label=category
            )
        
        # Set map extent
        if continental_states is not None:
            bounds = continental_states.total_bounds
            ax.set_xlim([bounds[0], bounds[2]])
            ax.set_ylim([bounds[1], bounds[3]])
        else:
            # Use alert bounds if state data not available
            bounds = alerts_gdf.total_bounds
            buffer = 100000  # 100km buffer in meters
            ax.set_xlim([bounds[0] - buffer, bounds[2] + buffer])
            ax.set_ylim([bounds[1] - buffer, bounds[3] + buffer])
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Add title and branding
        format_plot_title(ax, title, subtitle, font_props)
        
        # Create legend
        if risk_categories.size > 0:
            legend_elements = []
            for category in sorted(risk_categories):
                color = self.data_retriever.heat_risk_colors.get(category, COLORS['comparison'])
                legend_elements.append(
                    Patch(facecolor=color, edgecolor='white', alpha=0.7, label=category)
                )
            
            ax.legend(
                handles=legend_elements,
                loc='upper left',
                bbox_to_anchor=(0.02, 0.98),
                frameon=True,
                facecolor=COLORS['background'],
                edgecolor='#DDDDDD',
                fontsize=14,
                prop=font_props.get('regular') if font_props else None
            )
        
        # Add Deep Sky branding
        add_deep_sky_branding(
            ax, font_props, 
            data_note="DATA: NATIONAL WEATHER SERVICE"
        )
        
        # Save the map
        save_plot(fig, output_path)
        
        return fig
    
    def _create_empty_map(self, output_path: str, title: str, subtitle: str, 
                         figsize: Tuple[int, int]) -> plt.Figure:
        """
        Create an empty map when no heat alerts are available.
        
        Args:
            output_path (str): Path to save the map
            title (str): Map title
            subtitle (str): Map subtitle
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax, font_props = setup_enhanced_plot(figsize=figsize)
        
        # Load and plot US states
        try:
            states_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip')
            continental_states = states_gdf[~states_gdf['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
            continental_states = continental_states.to_crs(epsg=3857)
            
            continental_states.plot(ax=ax, color='#F0F0F0', edgecolor='#CCCCCC', linewidth=0.5)
            
            bounds = continental_states.total_bounds
            ax.set_xlim([bounds[0], bounds[2]])
            ax.set_ylim([bounds[1], bounds[3]])
            
        except Exception as e:
            print(f"Could not create base map: {e}")
        
        # Add "No Active Alerts" message
        ax.text(0.5, 0.5, 'NO ACTIVE HEAT ALERTS', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=24, fontweight='bold', color=COLORS['comparison'],
                fontproperties=font_props.get('bold') if font_props else None)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Add title and branding
        format_plot_title(ax, title, subtitle, font_props)
        add_deep_sky_branding(ax, font_props, data_note="DATA: NATIONAL WEATHER SERVICE")
        
        # Save the map
        save_plot(fig, output_path)
        
        return fig
    
    def get_all_heat_alerts(self) -> List[Dict]:
        """
        Get all current heat alerts, including those without geometry.
        
        Returns:
            List[Dict]: List of all heat alerts with risk categories and county FIPS codes
        """
        all_alerts = self.data_retriever.fetch_alerts()
        heat_alerts = []
        alerts_with_fips = 0
        
        for alert in all_alerts:
            if self.data_retriever.is_heat_alert(alert):
                properties = alert.get("properties", {})
                alert_data = {
                    'event_type': properties.get("event", ""),
                    'severity': properties.get("severity", ""),
                    'areas': properties.get("areaDesc", ""),
                    'effective': properties.get("effective", ""),
                    'expires': properties.get("expires", ""),
                    'headline': properties.get("headline", ""),
                    'description': properties.get("description", ""),
                    'sender_name': properties.get("senderName", "")
                }
                # Add risk categorization
                alert_data['risk_category'] = self.data_retriever.categorize_heat_risk(alert_data)
                
                # Add county FIPS codes if county matcher is available
                if self.county_matcher:
                    county_fips = self.county_matcher.get_county_fips_for_alert(alert)
                    alert_data['county_fips'] = county_fips
                    if county_fips:
                        alerts_with_fips += 1
                else:
                    alert_data['county_fips'] = []
                
                heat_alerts.append(alert_data)
        
        total_alerts = len(heat_alerts)
        coverage_pct = (alerts_with_fips / total_alerts * 100) if total_alerts > 0 else 0
        
        print(f"Found {total_alerts} total heat-related alerts")
        if self.county_matcher:
            print(f"Successfully mapped {alerts_with_fips} alerts to county FIPS codes ({coverage_pct:.1f}% coverage)")
        
        return heat_alerts
    
    def create_heat_alerts_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of current heat alerts.
        
        Returns:
            pd.DataFrame: Summary of heat alerts by risk category and location
        """
        heat_alerts = self.get_all_heat_alerts()
        
        if not heat_alerts:
            print("No heat alerts found")
            return pd.DataFrame()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(heat_alerts)
        
        # Create summary statistics
        summary = df.groupby(['risk_category', 'event_type']).agg({
            'areas': 'count',
            'severity': 'first'
        }).rename(columns={'areas': 'alert_count'}).reset_index()
        
        summary = summary.sort_values(['risk_category', 'alert_count'], ascending=[False, False])
        
        print(f"Heat Alert Summary:")
        print(summary.to_string(index=False))
        
        # Also create a detailed summary with specific areas and county FIPS codes
        columns_to_include = ['event_type', 'severity', 'risk_category', 'areas', 'effective', 'expires']
        if 'county_fips' in df.columns:
            columns_to_include.append('county_fips')
        if 'sender_name' in df.columns:
            columns_to_include.append('sender_name')
        
        detailed_summary = df[columns_to_include].copy()
        detailed_summary['areas_short'] = detailed_summary['areas'].str[:100] + '...'
        
        # Convert county_fips lists to string for CSV export
        if 'county_fips' in detailed_summary.columns:
            detailed_summary['county_fips_str'] = detailed_summary['county_fips'].apply(
                lambda x: ';'.join(x) if isinstance(x, list) else str(x)
            )
            detailed_summary['fips_count'] = detailed_summary['county_fips'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        detailed_summary = detailed_summary.sort_values(['risk_category', 'event_type'])
        
        return summary, detailed_summary
    
    def create_county_level_summary(self) -> pd.DataFrame:
        """
        Create a county-level summary with one row per county FIPS code.
        
        For counties with multiple alerts, keeps only the most severe alert based on:
        1. Event type priority: Extreme Heat Warning > Extreme Heat Watch > Heat Advisory
        2. Severity priority: Severe > Moderate > Minor
        
        Returns:
            pd.DataFrame: County-level summary with most severe alert per county
        """
        heat_alerts = self.get_all_heat_alerts()
        
        if not heat_alerts:
            print("No heat alerts found for county-level summary")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(heat_alerts)
        
        # Expand county FIPS codes - one row per county
        county_rows = []
        for _, alert in df.iterrows():
            county_fips_list = alert.get('county_fips', [])
            if county_fips_list:  # Only process alerts with valid FIPS codes
                for fips_code in county_fips_list:
                    county_row = {
                        'county_fips': fips_code,
                        'event_type': alert['event_type'],
                        'severity': alert['severity'],
                        'risk_category': alert['risk_category'],
                        'effective': alert['effective'],
                        'expires': alert['expires'],
                        'sender_name': alert.get('sender_name', ''),
                        'areas': alert['areas']
                    }
                    # Add priority score for ranking
                    county_row['priority_score'] = self.get_alert_priority_score(
                        alert['event_type'], alert['severity']
                    )
                    county_rows.append(county_row)
        
        if not county_rows:
            print("No valid county FIPS codes found in alerts")
            return pd.DataFrame()
        
        # Convert to DataFrame
        county_df = pd.DataFrame(county_rows)
        
        print(f"Expanded to {len(county_df)} county-alert combinations from {len(df)} alerts")
        
        # Find counties with multiple alerts
        counties_with_multiple = county_df['county_fips'].value_counts()
        multiple_alerts_count = (counties_with_multiple > 1).sum()
        if multiple_alerts_count > 0:
            print(f"Found {multiple_alerts_count} counties with multiple alerts - applying deduplication")
        
        # Keep only the highest priority alert per county
        # Sort by priority score (descending) and keep first occurrence of each county
        county_df_deduplicated = (county_df.sort_values(['county_fips', 'priority_score'], 
                                                       ascending=[True, False])
                                 .drop_duplicates(subset=['county_fips'], keep='first'))
        
        # Sort by priority score for final output (most severe alerts first)
        county_df_final = county_df_deduplicated.sort_values(['priority_score', 'county_fips'], 
                                                           ascending=[False, True])
        
        # Drop the priority score column for clean output
        county_df_final = county_df_final.drop(columns=['priority_score'])
        
        # Add county name and state information if county matcher is available
        if self.county_matcher and self.county_matcher.county_df is not None:
            county_info = self.county_matcher.county_df.copy()
            county_info['county_fips'] = county_info['combined_fips'].astype(str).str.zfill(5)
            
            # Merge county information - using RIGHT join to include ALL counties from reference data
            county_df_final = county_info[['county_fips', 'county_name', 'state_name']].merge(
                county_df_final, 
                on='county_fips', 
                how='left'  # Left join to keep all counties, fill NaN for counties without alerts
            )
            
            # Format timestamp columns for better readability
            county_df_final['effective_formatted'] = county_df_final['effective'].apply(self.format_timestamp)
            county_df_final['expires_formatted'] = county_df_final['expires'].apply(self.format_timestamp)
            
            # Add risk column based on event_type
            def map_event_to_risk(event_type):
                """Map event type to risk level."""
                if pd.isna(event_type):
                    return "None"
                elif event_type == "Extreme Heat Warning":
                    return "Severe"
                elif event_type == "Extreme Heat Watch":
                    return "High"
                elif event_type == "Heat Advisory":
                    return "Moderate"
                else:
                    return "None"  # Unknown event type
            
            county_df_final['risk'] = county_df_final['event_type'].apply(map_event_to_risk)
            
            # Fill empty/NaN values with "None" for counties without alerts
            alert_columns = ['event_type', 'severity', 'risk_category', 'effective', 'expires', 'sender_name']
            for col in alert_columns:
                county_df_final[col] = county_df_final[col].fillna("None")
            
            # Reorder columns to put location info first, with formatted timestamps and risk
            cols = ['county_fips', 'county_name', 'state_name', 'event_type', 'severity', 
                   'risk_category', 'risk', 'effective_formatted', 'expires_formatted', 'sender_name']
            county_df_final = county_df_final[cols]
            
            # Rename columns for cleaner output
            county_df_final = county_df_final.rename(columns={
                'effective_formatted': 'effective',
                'expires_formatted': 'expires'
            })
        
        initial_counties_with_alerts = len(county_df['county_fips'].unique())
        total_counties = len(county_df_final)
        counties_with_alerts = county_df_final['event_type'].notna().sum()
        counties_without_alerts = total_counties - counties_with_alerts
        
        print(f"County-level summary: {total_counties} total counties in output")
        print(f"  - {counties_with_alerts} counties with active heat alerts")
        print(f"  - {counties_without_alerts} counties with no active alerts")
        if multiple_alerts_count > 0:
            print(f"Deduplication: {initial_counties_with_alerts} counties reduced from {len(county_df)} total alerts")
        
        # Print sample of deduplication results
        if multiple_alerts_count > 0:
            sample_county = counties_with_multiple[counties_with_multiple > 1].index[0]
            original_alerts = county_df[county_df['county_fips'] == sample_county]
            final_alert = county_df_final[county_df_final['county_fips'] == sample_county]
            
            print(f"\nSample deduplication for county {sample_county}:")
            print(f"  Original alerts: {len(original_alerts)} ({', '.join(original_alerts['event_type'].tolist())})")
            if not final_alert.empty:
                print(f"  Final alert: {final_alert.iloc[0]['event_type']} ({final_alert.iloc[0]['severity']})")
        
        return county_df_final


def main():
    """Main function to run the heat wave mapper."""
    parser = argparse.ArgumentParser(description="Create heat wave risk maps from NWS alerts")
    parser.add_argument('-o', '--output', default='heat_wave_map.png', 
                       help='Output file path for the map')
    parser.add_argument('-t', '--title', default='CURRENT HEAT WAVE ALERTS',
                       help='Map title')
    parser.add_argument('-s', '--summary', action='store_true',
                       help='Print summary of heat alerts')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Print debug information about current alerts')
    parser.add_argument('-c', '--county-csv', 
                       default='/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/data/us_census/county_sq_miles.csv',
                       help='Path to county FIPS CSV file')
    parser.add_argument('--county-level', action='store_true',
                       help='Create county-level summary CSV with one row per county FIPS')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 16],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    print("Heat Wave Mapper - Deep Sky Research")
    print("=" * 50)
    
    # Initialize components
    data_retriever = HeatWaveDataRetriever()
    mapper = HeatWaveMapper(data_retriever, county_csv_path=args.county_csv)
    
    # Debug alert types if requested
    if args.debug:
        print("\nDebugging current alert types...")
        data_retriever.debug_alert_types()
        
        # Show sample FIPS mappings
        if mapper.county_matcher:
            print("\nSample county FIPS mappings:")
            heat_alerts = mapper.get_all_heat_alerts()
            for i, alert in enumerate(heat_alerts[:3]):  # Show first 3
                fips = alert.get('county_fips', [])
                areas = alert.get('areas', '')
                print(f"{i+1}. {areas[:80]}...")
                print(f"   FIPS: {fips}")
        
        if not args.summary:
            # If only debugging, don't create the map
            return
    
    # Create summary if requested
    if args.summary:
        result = mapper.create_heat_alerts_summary()
        if isinstance(result, tuple) and len(result) == 2:
            summary_df, detailed_df = result
            if not summary_df.empty:
                print("\nSaving heat alerts summary to heat_alerts_summary.csv")
                summary_df.to_csv('heat_alerts_summary.csv', index=False)
                print("Saving detailed heat alerts to heat_alerts_detailed.csv")
                detailed_df.to_csv('heat_alerts_detailed.csv', index=False)
        elif not result.empty if hasattr(result, 'empty') else False:
            print("\nSaving heat alerts summary to heat_alerts_summary.csv")
            result.to_csv('heat_alerts_summary.csv', index=False)
    
    # Create county-level summary if requested
    if args.county_level:
        print("\nCreating county-level summary...")
        county_df = mapper.create_county_level_summary()
        if not county_df.empty:
            print("Saving county-level summary to heat_alerts_by_county.csv")
            county_df.to_csv('heat_alerts_by_county.csv', index=False)
            
            # Show sample of results
            print(f"\nSample county-level results (first 5 rows):")
            if len(county_df) > 0:
                print(county_df.head().to_string(index=False))
        else:
            print("No county-level data to save")
    
    # Create heat risk map only if not in debug-only mode and not county-level-only mode
    if not (args.debug and not args.summary) and not (args.county_level and not args.summary):
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating heat risk map...")
        fig = mapper.create_heat_risk_map(
            output_path=args.output,
            title=args.title,
            figsize=tuple(args.figsize)
        )
        
        print(f"Heat wave map saved to: {args.output}")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()