"""
Dynamic legend generation for Folium maps

Creates elegant, responsive legends that update based on active data layers.
"""

from typing import Dict, List

import folium
from branca.element import MacroElement, Template


def create_dynamic_legend(layer_data: Dict) -> MacroElement:
    """
    Create a dynamic HTML/CSS legend for the map.

    Parameters:
    -----------
    layer_data : dict
        Dictionary with layer keys and their config/data
        Format: {layer_key: {'data': data, 'config': layer_config}}

    Returns:
    --------
    MacroElement: Legend element to add to Folium map
    """

    # Build legend items from active layers
    legend_items = []

    for layer_key, layer_info in layer_data.items():
        config = layer_info["config"]
        category = layer_key.split(".")[0]

        if category == "storage":
            # Check if this is a graduated color layer (has thresholds)
            if "thresholds" in config and "colors" in config:
                # Sedimentary thickness: gradient with thresholds
                thresholds = config["thresholds"]
                colors = config["colors"]
                legend_items.append(
                    {
                        "type": "gradient",
                        "label": config["name"],
                        "items": [
                            {
                                "color": colors[i],
                                "label": f"{thresholds[i]:,.0f}-{thresholds[i + 1]:,.0f}"
                                if i < len(thresholds) - 1
                                else f"≥{thresholds[i]:,.0f}",
                            }
                            for i in range(len(colors))
                        ],
                        "unit": "meters",
                    }
                )
            else:
                # Storage layers: simple color box with name
                legend_items.append(
                    {
                        "type": "color",
                        "color": config["color"],
                        "label": config["name"],
                        "description": config.get("description", ""),
                    }
                )

        elif category == "energy":
            layer_name = layer_key.split(".")[1]

            if layer_name == "solar":
                # Solar: gradient with thresholds
                thresholds = config.get("thresholds", [])
                colors = config.get("colors", [])
                legend_items.append(
                    {
                        "type": "gradient",
                        "label": config["name"],
                        "items": [
                            {
                                "color": colors[i],
                                "label": f"{thresholds[i]:.1f}-{thresholds[i + 1]:.1f}"
                                if i < len(thresholds) - 1
                                else f"≥{thresholds[i]:.1f}",
                            }
                            for i in range(len(colors))
                        ],
                        "unit": "kWh/kWp/day",
                    }
                )

            elif layer_name == "geothermal":
                # Geothermal: depth categories
                thresholds = config.get("thresholds", [4, 6, 8])
                colors = config.get("colors", [])
                depth_labels = [
                    f"≤{thresholds[0]} km",
                    f"{thresholds[0]}-{thresholds[1]} km",
                    f"{thresholds[1]}-{thresholds[2]} km",
                    f">{thresholds[2]} km",
                ]
                legend_items.append(
                    {
                        "type": "gradient",
                        "label": config["name"],
                        "items": [
                            {"color": colors[i], "label": depth_labels[i]}
                            for i in range(len(colors))
                        ],
                        "unit": "Drilling depth to 150°C",
                    }
                )

    # Generate HTML
    html = _generate_legend_html(legend_items)

    # Create MacroElement with the HTML template
    legend = MacroElement()
    legend._template = Template(html)

    return legend


def _generate_legend_html(legend_items: List[Dict]) -> str:
    """
    Generate HTML/CSS for the legend.

    Parameters:
    -----------
    legend_items : list
        List of legend item dictionaries

    Returns:
    --------
    str: HTML template string
    """

    # Build legend content
    items_html = []

    for item in legend_items:
        if item["type"] == "color":
            # Simple color box with label
            items_html.append(f"""
                <div class="legend-item">
                    <div class="legend-color-box" style="background-color: {item["color"]};"></div>
                    <div class="legend-label">
                        <div class="legend-title">{item["label"]}</div>
                        <div class="legend-description">{item["description"]}</div>
                    </div>
                </div>
            """)

        elif item["type"] == "gradient":
            # Gradient with multiple values
            gradient_items = []
            for sub_item in item["items"]:
                gradient_items.append(f"""
                    <div class="legend-gradient-item">
                        <div class="legend-color-box" style="background-color: {sub_item["color"]};"></div>
                        <span>{sub_item["label"]}</span>
                    </div>
                """)

            items_html.append(f"""
                <div class="legend-section">
                    <div class="legend-section-title">{item["label"]}</div>
                    <div class="legend-section-unit">{item.get("unit", "")}</div>
                    {"".join(gradient_items)}
                </div>
            """)

    # Complete HTML template
    html = f"""
    {{% macro html(this, kwargs) %}}
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        .legend {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.5;
            max-width: 280px;
            z-index: 1000;
        }}

        .legend-header {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 12px;
            color: #2c3e50;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 10px;
            gap: 10px;
        }}

        .legend-color-box {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            flex-shrink: 0;
            border: 1px solid rgba(0,0,0,0.1);
        }}

        .legend-label {{
            flex: 1;
        }}

        .legend-title {{
            font-weight: 500;
            color: #2c3e50;
            margin-bottom: 2px;
        }}

        .legend-description {{
            font-size: 11px;
            color: #7f8c8d;
            line-height: 1.3;
        }}

        .legend-section {{
            margin-bottom: 15px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}

        .legend-section:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}

        .legend-section-title {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
            font-size: 13px;
        }}

        .legend-section-unit {{
            font-size: 11px;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}

        .legend-gradient-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 12px;
            color: #34495e;
        }}

        .legend-gradient-item:last-child {{
            margin-bottom: 0;
        }}

        /* Responsive design for smaller screens */
        @media (max-width: 768px) {{
            .legend {{
                bottom: 10px;
                right: 10px;
                max-width: 200px;
                font-size: 11px;
                padding: 12px 15px;
            }}

            .legend-color-box {{
                width: 16px;
                height: 16px;
            }}
        }}
    </style>
    </head>
    <body>
        <div class="legend">
            <div class="legend-header">Data Layers</div>
            {"".join(items_html)}
        </div>
    </body>
    </html>
    {{% endmacro %}}
    """

    return html


def add_legend_to_map(m: folium.Map, layer_data: Dict) -> folium.Map:
    """
    Add a dynamic legend to a Folium map.

    Parameters:
    -----------
    m : folium.Map
        The map to add the legend to
    layer_data : dict
        Dictionary with layer keys and their config/data

    Returns:
    --------
    folium.Map: Map with legend added
    """
    if not layer_data:
        # No layers selected, don't add legend
        return m

    legend = create_dynamic_legend(layer_data)
    m.get_root().add_child(legend)

    return m
