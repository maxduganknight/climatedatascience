import requests
import json
from datetime import datetime, timezone
import time
import os
from typing import List, Dict, Set
import argparse
import sys

sys.path.append('/Users/max/Deep_Sky/')
from creds import NWS_WEB_HOOK_URL, NWS_TEST_WEB_HOOK_URL, NWS_WRITE_TOKEN, NWS_OAUTH_TOKEN

from helpers import send_slack_message

class WeatherAlertMonitor:
    def __init__(self, slack_webhook_url: str):
        self.slack_webhook_url = slack_webhook_url
        self.nws_alerts_url = "https://api.weather.gov/alerts/active"
        self.sent_alerts = set()  # Track sent alerts to avoid duplicates
        
        # Define extreme weather event types to monitor
        self.extreme_events = {
            # "Tornado Warning",
            "Extreme Fire Danger",
            "Hurricane Warning",
            # "Hurricane Watch",
            "Hurricane Force Wind Warning",
            # "Hurricane Force Wind Watch",
            "Tropical Storm Warning",
            # "Tropical Storm Watch",
            # "Tropical Cyclone Local Statement",
            # "Ice Storm Warning",
            # "Extreme Wind Warning",
            "Storm Surge Warning",
            # "Extreme Heat Warning",
            # "Fire Weather Watch",
            # "Fire Warning",
            "Red Flag Warning",
            # "Low Water Advisory"
        }
    
    def fetch_alerts(self) -> List[Dict]:
        """Fetch current alerts from NWS API"""
        try:
            response = requests.get(
                self.nws_alerts_url,
                headers={"User-Agent": "WeatherAlertBot/1.0"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("features", [])
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []
    
    def is_extreme_alert(self, alert: Dict) -> bool:
        """Check if alert is for extreme weather"""
        properties = alert.get("properties", {})
        event_type = properties.get("event", "")
        severity = properties.get("severity", "")
        return event_type in self.extreme_events and severity in ["Extreme", "Severe"]
    
    def format_alert_message(self, alert: Dict) -> str:
        """Format alert for Slack message"""
        props = alert["properties"]
        # Extract key information
        event = props.get("event", "Unknown")
        severity = props.get("severity", "Unknown")
        urgency = props.get("urgency", "Unknown")
        areas = props.get("areaDesc", "Unknown location")
        headline = props.get("headline", "")
        description = props.get("description", "")
        
        # Extract time information
        sent_time = props.get("sent", "")
        if sent_time:
            try:
                # Parse ISO format and convert to readable format
                dt = datetime.fromisoformat(sent_time.replace('Z', '+00:00'))
                alert_time = dt.strftime("%b %d at %I:%M %p")
            except:
                alert_time = sent_time
        else:
            alert_time = "Unknown"
        
        # Create clean, formatted message
        message = f"New NWS Alert: {event}\n"
        message += f"Location: {areas}\n"
        message += f"Severity: {severity} | Urgency: {urgency}\n"
        message += f"Alert Time: {alert_time}\n"
        
        if headline:
            message += f"Summary: {headline}\n"
        
        # Add condensed details (first 200 chars)
        if description:
            clean_desc = description.replace('\n', ' ').strip()[:200]
            if len(description) > 200:
                clean_desc += "..."
            message += f"Details: {clean_desc}\n\n"
        
        return message

    
    def process_alerts(self):
        """Fetch and process new alerts"""
        alerts = self.fetch_alerts()
        new_alerts_count = 0
        
        for alert in alerts:
            if not self.is_extreme_alert(alert):
                continue
            
            # Use alert ID to track if we've already sent this
            alert_id = alert.get("properties", {}).get("id", "")
            if alert_id in self.sent_alerts:
                continue
            
            # Format and send the alert
            message = self.format_alert_message(alert)
            try:
                send_slack_message(self.slack_webhook_url, message)
                self.sent_alerts.add(alert_id)
                new_alerts_count += 1
                print(f"Sent alert: {alert['properties']['event']} - {alert['properties']['areaDesc']}")
            except Exception as e:
                print(f"Failed to send Slack message: {e}")
        
        # Clean up old alerts (keep only last 1000)
        if len(self.sent_alerts) > 1000:
            self.sent_alerts = set(list(self.sent_alerts)[-1000:])
        
        return new_alerts_count
    
    def check_and_send_alerts(self):
        """Check for alerts and send them once"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Checking for extreme weather alerts...")
        
        new_alerts = self.process_alerts()
        if new_alerts > 0:
            print(f"Found and sent {new_alerts} new extreme weather alerts")
        else:
            print("No new extreme weather alerts found")
        
        return new_alerts


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NWS extreme weather alerts and send to Slack")
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode with test webhook URL')
    parser.add_argument('-s', '--slack', action='store_true', help='Send a Slack message')
    args = parser.parse_args()

    if args.test:
        print("Running in test mode with test webhook URL")
        slack_webhook = NWS_TEST_WEB_HOOK_URL
    else:
        print("Running in production mode with live webhook URL")
        slack_webhook = NWS_WEB_HOOK_URL
    if not slack_webhook:
        print("Please set NWS_WEB_HOOK_URL environment variable")
        print("Get webhook URL from: https://api.slack.com/messaging/webhooks")
    else:
        monitor = WeatherAlertMonitor(slack_webhook_url=slack_webhook)
        monitor.check_and_send_alerts()