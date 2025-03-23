import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import folium
from folium.plugins import HeatMap
import webbrowser
import os
import json
from branca.colormap import LinearColormap
import matplotlib.colors as mcolors


# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Convert datetime to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek

    return df


# Function to define health risk categories based on AQI
def get_health_risk(aqi):
    if aqi <= 50:
        return "Good - No health risks"
    elif aqi <= 100:
        return "Moderate - Sensitive individuals may experience respiratory symptoms"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups - Children and elderly may experience health effects"
    elif aqi <= 200:
        return "Unhealthy - Everyone may begin to experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy - Health warnings of emergency conditions"
    else:
        return "Hazardous - Everyone may experience more serious health effects"


# Function to get color for AQI level
def get_aqi_color(aqi):
    if aqi <= 50:
        return "#00E400"  # Green
    elif aqi <= 100:
        return "#FFFF00"  # Yellow
    elif aqi <= 150:
        return "#FF7E00"  # Orange
    elif aqi <= 200:
        return "#FF0000"  # Red
    elif aqi <= 300:
        return "#99004C"  # Purple
    else:
        return "#7E0023"  # Maroon


# Function to train the prediction model
def train_prediction_model(df):
    # Features for prediction
    features = ['components.co', 'components.no', 'components.no2',
                'components.o3', 'components.so2', 'components.pm2_5',
                'components.pm10', 'components.nh3', 'hour', 'day',
                'month', 'day_of_week']

    X = df[features]
    y = df['main.aqi']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Calculate and print accuracy
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Training score: {train_score:.4f}")
    print(f"Testing score: {test_score:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return model, scaler, features


# Function to predict future AQI
# Function to predict future AQI
def predict_future_aqi(model, scaler, features, df, future_days=7):
    # Get the last date in the dataset
    last_date = df['datetime'].max()

    # Create a dataframe for future dates
    future_dates = [last_date + timedelta(days=i + 1) for i in range(future_days)]

    # For each city, predict future AQI
    cities = df['city_name'].unique()
    all_predictions = []

    for city in cities:
        # Get the most recent data points for this city (last 3 days)
        city_recent_data = df[df['city_name'] == city].sort_values('datetime').tail(72)  # 3 days x 24 hours

        # If less than 3 days of data, use what's available
        if city_recent_data.empty:
            continue

        # Get the latest data point as a starting reference
        latest_city_data = city_recent_data.iloc[-1:].copy()

        # For each future date
        for date in future_dates:
            # Create a new row based on the latest data
            new_row = latest_city_data.copy()
            new_row['datetime'] = date

            # Update time-based features
            new_row['hour'] = date.hour
            new_row['day'] = date.day
            new_row['month'] = date.month
            new_row['day_of_week'] = date.dayofweek

            # Add more variability to pollution components
            for component in ['components.co', 'components.no', 'components.no2',
                              'components.o3', 'components.so2', 'components.pm2_5',
                              'components.pm10', 'components.nh3']:

                if len(city_recent_data) > 1:
                    # Calculate recent trend with more variability
                    recent_values = city_recent_data[component].values
                    avg_change = np.mean(np.diff(recent_values[-24:])) if len(recent_values) >= 24 else 0

                    # Apply trend with more random variation
                    current_value = new_row[component].values[0]
                    new_value = current_value + (avg_change * np.random.uniform(1.0, 2.0)) + np.random.normal(0,
                                                                                                              current_value * 0.2)

                    # Ensure values don't go negative
                    new_row[component] = max(0.1, new_value)

            # Simulate weekday vs. weekend patterns
            if date.dayofweek >= 5:  # Weekend
                new_row['components.pm2_5'] *= np.random.uniform(0.8, 1.2)  # Random variation for weekends
                new_row['components.pm10'] *= np.random.uniform(0.8, 1.2)
            else:  # Weekday
                new_row['components.pm2_5'] *= np.random.uniform(1.0, 1.5)  # Higher pollution on weekdays
                new_row['components.pm10'] *= np.random.uniform(1.0, 1.5)

            # Simulate occasional extreme events (e.g., wildfires, industrial accidents)
            if np.random.rand() < 0.1:  # 10% chance of an extreme event
                new_row['components.pm2_5'] *= np.random.uniform(2.0, 5.0)
                new_row['components.pm10'] *= np.random.uniform(2.0, 5.0)

            # Prepare features for prediction
            X_pred = new_row[features]
            X_pred_scaled = scaler.transform(X_pred)

            # Predict AQI
            predicted_aqi = model.predict(X_pred_scaled)[0]
            new_row['main.aqi'] = round(predicted_aqi)
            new_row['predicted'] = True
            new_row['health_risk'] = get_health_risk(predicted_aqi)

            all_predictions.append(new_row)

            # Update the latest data to the newly predicted row for the next prediction
            latest_city_data = new_row.copy()

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    return predictions_df

# Function to get health recommendations
def get_recommendations(aqi):
    if aqi <= 50:
        return "Air quality is good. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Children, elderly, and individuals with respiratory conditions should limit outdoor activities."
    elif aqi <= 200:
        return "Everyone should avoid prolonged outdoor exertion. Wear masks when outdoors."
    elif aqi <= 300:
        return "Everyone should avoid outdoor activities. Close windows to reduce indoor pollution."
    else:
        return "EMERGENCY CONDITIONS: Everyone should stay indoors with purifiers if possible. Wear N95 masks if going outdoors is necessary."


# Generate synthetic data for demonstration
def generate_synthetic_data(df):
    # Define cities with coordinates
    cities_data = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "base_aqi": np.random.randint(1, 5)},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "base_aqi": np.random.randint(2, 6)},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "base_aqi": np.random.randint(1, 4)},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "base_aqi": np.random.randint(2, 5)},
        {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "base_aqi": np.random.randint(3, 6)},
        {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652, "base_aqi": np.random.randint(1, 4)},
        {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936, "base_aqi": np.random.randint(1, 3)},
        {"name": "San Diego", "lat": 32.7157, "lon": -117.1611, "base_aqi": np.random.randint(1, 4)},
        {"name": "Dallas", "lat": 32.7767, "lon": -96.7970, "base_aqi": np.random.randint(2, 5)},
        {"name": "San Jose", "lat": 37.3382, "lon": -121.8863, "base_aqi": np.random.randint(1, 4)},
        {"name": "Austin", "lat": 30.2672, "lon": -97.7431, "base_aqi": np.random.randint(1, 3)},
        {"name": "Jacksonville", "lat": 30.3322, "lon": -81.6557, "base_aqi": np.random.randint(1, 3)},
        {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194, "base_aqi": np.random.randint(1, 3)},
        {"name": "Indianapolis", "lat": 39.7684, "lon": -86.1581, "base_aqi": np.random.randint(2, 4)},
        {"name": "Seattle", "lat": 47.6062, "lon": -122.3321, "base_aqi": np.random.randint(1, 3)}
    ]

    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='H')

    # Create synthetic records
    synthetic_data = []

    for city_info in cities_data:
        base_aqi = city_info["base_aqi"]
        for date in dates:
            # Add some realistic variation
            hour_effect = np.sin(date.hour / 24 * 2 * np.pi) * 1.5  # Higher in morning and evening
            day_effect = (date.day % 7) / 7  # Weekly cycle

            aqi = max(1, round(base_aqi + hour_effect + day_effect + np.random.normal(0, 0.5)))

            # Create components based on AQI
            co = np.random.uniform(300, 600)
            no = np.random.uniform(0, 0.1)
            no2 = np.random.uniform(2, 12) * aqi / 3
            o3 = np.random.uniform(10, 70) * aqi / 3
            so2 = np.random.uniform(0.5, 2.5)
            pm2_5 = np.random.uniform(5, 15) * aqi / 3
            pm10 = pm2_5 * np.random.uniform(1.1, 1.4)
            nh3 = np.random.uniform(1.5, 4.5)

            row = {
                'datetime': date,
                'main.aqi': aqi,
                'components.co': co,
                'components.no': no,
                'components.no2': no2,
                'components.o3': o3,
                'components.so2': so2,
                'components.pm2_5': pm2_5,
                'components.pm10': pm10,
                'components.nh3': nh3,
                'city_name': city_info["name"],
                'lat': city_info["lat"],
                'lon': city_info["lon"]
            }

            synthetic_data.append(row)

    return pd.DataFrame(synthetic_data)


# Create a geospatial map
def create_map(df, predictions_df=None, date_filter=None):
    # Center the map on the average of all coordinates
    avg_lat = df['lat'].mean()
    avg_lon = df['lon'].mean()

    # Create a base map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles='OpenStreetMap')

    # If we have predictions and a specific date filter
    if predictions_df is not None and date_filter is not None:
        # Filter predictions for the specified date
        day_predictions = predictions_df[predictions_df['datetime'].dt.date == date_filter.date()]

        # Add markers for each city with predictions
        for city in day_predictions['city_name'].unique():
            city_data = day_predictions[day_predictions['city_name'] == city].iloc[0]
            aqi_value = city_data['main.aqi']
            color = get_aqi_color(aqi_value)

            # Create popup content
            popup_content = f"""
            <div style="width:200px;">
                <h4>{city}</h4>
                <p><b>Date:</b> {city_data['datetime'].strftime('%Y-%m-%d')}</p>
                <p><b>AQI:</b> {aqi_value}</p>
                <p><b>Health Risk:</b> {city_data['health_risk'].split(' - ')[0]}</p>
            </div>
            """

            # Add marker
            folium.CircleMarker(
                location=[city_data['lat'], city_data['lon']],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)
    else:
        # Get the latest measurement for each city
        latest_data = df.sort_values('datetime').groupby('city_name').last().reset_index()

        # Add markers for each city
        for _, city_data in latest_data.iterrows():
            aqi_value = city_data['main.aqi']
            color = get_aqi_color(aqi_value)

            # Create popup content
            popup_content = f"""
            <div style="width:200px;">
                <h4>{city_data['city_name']}</h4>
                <p><b>Date:</b> {city_data['datetime'].strftime('%Y-%m-%d %H:%M')}</p>
                <p><b>AQI:</b> {aqi_value}</p>
                <p><b>Health Risk:</b> {get_health_risk(aqi_value).split(' - ')[0]}</p>
            </div>
            """

            # Add marker
            folium.CircleMarker(
                location=[city_data['lat'], city_data['lon']],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)

    # Add legend
    colormap = LinearColormap(
        ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#99004C', '#7E0023'],
        vmin=0, vmax=300,
        caption='Air Quality Index (AQI)'
    )
    m.add_child(colormap)

    # Save the map to an HTML file
    map_path = 'air_quality_map.html'
    m.save(map_path)

    return map_path


# Create a heatmap of air quality
def create_heatmap(df, predictions_df=None, date_filter=None):
    # Center the map on the average of all coordinates
    avg_lat = df['lat'].mean()
    avg_lon = df['lon'].mean()

    # Create a base map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles='OpenStreetMap')

    # Prepare data for heatmap
    if predictions_df is not None and date_filter is not None:
        # Filter predictions for the specified date
        heat_data = predictions_df[predictions_df['datetime'].dt.date == date_filter.date()]
    else:
        # Use latest data for each city
        heat_data = df.sort_values('datetime').groupby('city_name').last().reset_index()

    # Create heat data points
    heat_points = []
    for _, row in heat_data.iterrows():
        # Weight by AQI - higher AQI = hotter point
        heat_points.append([row['lat'], row['lon'], row['main.aqi'] / 10])  # Scale down AQI for better visualization

    # Add heatmap to the map
    HeatMap(heat_points, radius=25).add_to(m)

    # Add regular markers with city names for reference
    for _, row in heat_data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['city_name'],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 10pt; color: black; font-weight: bold; 
                background-color: white; border-radius: 3px; padding: 1px 3px;">
                    {row['city_name']}
                </div>
            """)
        ).add_to(m)

    # Save the map to an HTML file
    heatmap_path = 'air_quality_heatmap.html'
    m.save(heatmap_path)

    return heatmap_path


# Function to create and run the UI application
def run_application(df, predictions_df):
    # Prepare data for visualization
    df['predicted'] = False
    df['health_risk'] = df['main.aqi'].apply(get_health_risk)

    predictions_df['predicted'] = True
    predictions_df['health_risk'] = predictions_df['main.aqi'].apply(get_health_risk)

    combined_df = pd.concat([df, predictions_df], ignore_index=True)
    cities = combined_df['city_name'].unique()

    # Generate health advisory
    advisory_data = []
    for city in predictions_df['city_name'].unique():
        city_data = predictions_df[predictions_df['city_name'] == city]
        max_aqi = city_data['main.aqi'].max()
        avg_aqi = city_data['main.aqi'].mean()
        worst_day = city_data.loc[city_data['main.aqi'].idxmax()]

        advisory = {
            'city': city,
            'average_aqi': round(avg_aqi, 2),
            'max_aqi': max_aqi,
            'max_aqi_date': worst_day['datetime'].strftime('%Y-%m-%d'),
            'worst_health_risk': get_health_risk(max_aqi),
            'recommendations': get_recommendations(max_aqi)
        }

        advisory_data.append(advisory)

    advisory_df = pd.DataFrame(advisory_data)

    # Create UI application
    root = tk.Tk()
    root.title("Air Quality Prediction and Health Risk Assessment")
    root.geometry("1200x800")  # Increased size for better visualization with many cities

    # Create tabs
    tab_control = ttk.Notebook(root)

    # City prediction tab
    city_tab = ttk.Frame(tab_control)
    tab_control.add(city_tab, text='City Predictions')

    # Health risk tab
    risk_tab = ttk.Frame(tab_control)
    tab_control.add(risk_tab, text='Health Risk Summary')

    # Advisory tab
    advisory_tab = ttk.Frame(tab_control)
    tab_control.add(advisory_tab, text='Health Advisory')

    # Map tab
    map_tab = ttk.Frame(tab_control)
    tab_control.add(map_tab, text='Geographic Map')

    # Heatmap tab
    heatmap_tab = ttk.Frame(tab_control)
    tab_control.add(heatmap_tab, text='Heat Map')

    # Spatial analysis tab
    spatial_tab = ttk.Frame(tab_control)
    tab_control.add(spatial_tab, text='Spatial Analysis')

    tab_control.pack(expand=1, fill='both')

    # City prediction tab content
    city_frame = ttk.Frame(city_tab)
    city_frame.pack(pady=10)

    city_label = ttk.Label(city_frame, text="Select City:")
    city_label.grid(row=0, column=0, padx=5, pady=5)

    city_var = tk.StringVar()
    city_dropdown = ttk.Combobox(city_frame, textvariable=city_var, values=list(cities))
    city_dropdown.grid(row=0, column=1, padx=5, pady=5)
    city_dropdown.current(0)

    # Figure frame
    fig_frame = ttk.Frame(city_tab)
    fig_frame.pack(pady=10, fill='both', expand=True)

    # Create figure
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    # Create canvas for figure
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill='both', expand=True)

    # Navigation buttons frame
    nav_frame = ttk.Frame(city_tab)
    nav_frame.pack(pady=10)

    current_city_index = 0

    def update_city_plot():
        ax.clear()
        selected_city = city_var.get()

        # Get data for the selected city
        city_data = combined_df[combined_df['city_name'] == selected_city]

        # Ensure data is sorted by datetime for proper plotting
        city_data = city_data.sort_values('datetime')

        # Verify predictions are properly flagged
        historical = city_data[city_data['predicted'] == False]
        predicted = city_data[city_data['predicted'] == True]

        # If predicted data is empty, check if we need to fix the flag
        if predicted.empty and not predictions_df.empty:
            # Get predictions for this city
            city_predictions = predictions_df[predictions_df['city_name'] == selected_city]
            if not city_predictions.empty:
                # Plot the predictions from predictions_df directly
                ax.plot(city_predictions['datetime'], city_predictions['main.aqi'], 'r--', label='Predicted AQI')
                ax.scatter(city_predictions['datetime'], city_predictions['main.aqi'], color='red', s=30)
        else:
            # Plot historical data
            if not historical.empty:
                ax.plot(historical['datetime'], historical['main.aqi'], 'b-', label='Historical AQI')

            # Plot predicted data
            if not predicted.empty:
                ax.plot(predicted['datetime'], predicted['main.aqi'], 'r--', label='Predicted AQI')
                # Add points to the predicted line to make it more visible
                ax.scatter(predicted['datetime'], predicted['main.aqi'], color='red', s=30)

        # Set title with wrapping to prevent overflow
        title = f'Air Quality Index (AQI) for {selected_city}'
        ax.set_title(title)

        ax.set_xlabel('Date')
        ax.set_ylabel('AQI')
        ax.legend()
        ax.grid(True)

        # Add vertical line to separate historical and predicted data
        if not historical.empty and (
                not predicted.empty or not predictions_df[predictions_df['city_name'] == selected_city].empty):
            separation_date = historical['datetime'].max() + timedelta(hours=1)
            ax.axvline(x=separation_date, color='gray', linestyle='--')
            ax.text(separation_date, ax.get_ylim()[1] * 0.95, 'Prediction Start',
                    rotation=90, verticalalignment='top')

        fig.tight_layout()
        canvas.draw()

    def next_city():
        nonlocal current_city_index
        current_city_index = (current_city_index + 1) % len(cities)
        city_var.set(cities[current_city_index])
        update_city_plot()

    def prev_city():
        nonlocal current_city_index
        current_city_index = (current_city_index - 1) % len(cities)
        city_var.set(cities[current_city_index])
        update_city_plot()

    prev_button = ttk.Button(nav_frame, text="Previous City", command=prev_city)
    prev_button.grid(row=0, column=0, padx=5, pady=5)

    next_button = ttk.Button(nav_frame, text="Next City", command=next_city)
    next_button.grid(row=0, column=1, padx=5, pady=5)

    # Health risk tab content
    # Create a frame to hold all health risk visualization elements
    risk_main_frame = ttk.Frame(risk_tab)
    risk_main_frame.pack(pady=10, fill='both', expand=True)

    # Create a title frame at the top
    risk_title_frame = ttk.Frame(risk_main_frame)
    risk_title_frame.pack(side=tk.TOP, fill=tk.X)

    # Add title label with custom styling
    title_label = ttk.Label(
        risk_title_frame,
        text="Health Risk Forecast by City",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=5)

    # Create a frame for the plot
    risk_plot_frame = ttk.Frame(risk_main_frame)
    risk_plot_frame.pack(side=tk.LEFT, fill='both', expand=True)

    # Create a frame for the legend
    legend_frame = ttk.Frame(risk_main_frame)
    legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

    # Add legend title
    legend_title = ttk.Label(legend_frame, text="Health Risk Categories", font=("Arial", 12, "bold"))
    legend_title.pack(pady=(0, 10))

    # Define risk categories and colors
    risk_colors = {
        "Good - No health risks": "#00E400",  # Green
        "Moderate - Sensitive individuals may experience respiratory symptoms": "#FFFF00",  # Yellow
        "Unhealthy for Sensitive Groups - Children and elderly may experience health effects": "#FF7E00",  # Orange
        "Unhealthy - Everyone may begin to experience health effects": "#FF0000",  # Red
        "Very Unhealthy - Health warnings of emergency conditions": "#99004C",  # Purple
        "Hazardous - Everyone may experience more serious health effects": "#7E0023"  # Maroon
    }

    # Create custom legend items manually
    for risk, color in risk_colors.items():
        # Create a frame for each legend entry
        entry_frame = ttk.Frame(legend_frame)
        entry_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.W)

        # Create a colored box
        color_label = tk.Label(entry_frame, bg=color, width=2, height=1)
        color_label.pack(side=tk.LEFT, padx=(0, 5))

        # Create shortened risk text
        short_risk = risk.split(" - ")[0]
        text_label = ttk.Label(entry_frame, text=short_risk)
        text_label.pack(side=tk.LEFT, anchor=tk.W)

    # Create figure for bar chart
    risk_fig = Figure(figsize=(9, 5))  # Wider figure for more cities
    risk_ax = risk_fig.add_subplot(111)

    # Create canvas for figure
    risk_canvas = FigureCanvasTkAgg(risk_fig, master=risk_plot_frame)
    risk_canvas_widget = risk_canvas.get_tk_widget()
    risk_canvas_widget.pack(fill='both', expand=True)

    # Add scroll control frame
    scroll_frame = ttk.Frame(risk_plot_frame)
    scroll_frame.pack(fill=tk.X, side=tk.BOTTOM)

    # Add scrollbar and controls
    scrollbar_label = ttk.Label(scroll_frame, text="Scroll cities:")
    scrollbar_label.pack(side=tk.LEFT, padx=5)

    # Add city range variable
    city_start_index = 0
    max_cities_per_view = 15  # Show this many cities at once

    def update_risk_plot():
        risk_ax.clear()

        # Get number of cities to determine display approach
        nonlocal city_start_index
        city_count = len(predictions_df['city_name'].unique())

        # Create health risk summary
        risk_summary = predictions_df.groupby(['city_name', 'health_risk']).size().reset_index(name='days')

        # Filter cities for display based on current view
        if city_count > max_cities_per_view:
            end_index = min(city_start_index + max_cities_per_view, city_count)
            visible_cities = sorted(predictions_df['city_name'].unique())[city_start_index:end_index]
            filtered_summary = risk_summary[risk_summary['city_name'].isin(visible_cities)]

            # Update scrollbar status text
            scroll_status.config(text=f"Showing {city_start_index + 1}-{end_index} of {city_count} cities")
        else:
            filtered_summary = risk_summary
            scroll_status.config(text=f"Showing all {city_count} cities")

        # Create horizontal bar chart for better readability with many cities
        bars = sns.barplot(
            x='days',
            y='city_name',
            hue='health_risk',
            data=filtered_summary,
            ax=risk_ax,
            palette=risk_colors,
            orient='h'  # Horizontal bars
        )

        # Remove the legend as we have a custom one
        risk_ax.get_legend().remove()

        # Set labels
        risk_ax.set_xlabel('Number of Days')
        risk_ax.set_ylabel('City')

        # Add a subtitle to show measurement period
        min_date = predictions_df['datetime'].min().strftime('%Y-%m-%d')
        max_date = predictions_df['datetime'].max().strftime('%Y-%m-%d')
        risk_ax.text(
            0.5, 1.02,
            f"Forecast Period: {min_date} to {max_date}",
            horizontalalignment='center',
            transform=risk_ax.transAxes,
            fontsize=10
        )

        # Add grid lines for better readability
        risk_ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Adjust layout
        risk_fig.tight_layout()
        risk_canvas.draw()

    def scroll_next():
        nonlocal city_start_index
        city_count = len(predictions_df['city_name'].unique())
        if city_start_index + max_cities_per_view < city_count:
            city_start_index += max_cities_per_view
            update_risk_plot()

    def scroll_prev():
        nonlocal city_start_index
        if city_start_index >= max_cities_per_view:
            city_start_index -= max_cities_per_view
        else:
            city_start_index = 0
        update_risk_plot()

    # Add scroll buttons
    prev_scroll_btn = ttk.Button(scroll_frame, text="◀ Previous", command=scroll_prev)
    prev_scroll_btn.pack(side=tk.LEFT, padx=5)

    scroll_status = ttk.Label(scroll_frame, text="", width=25)
    scroll_status.pack(side=tk.LEFT, padx=5)

    next_scroll_btn = ttk.Button(scroll_frame, text="Next ▶", command=scroll_next)
    next_scroll_btn.pack(side=tk.LEFT, padx=5)

    # Advisory tab content
    advisory_frame = ttk.Frame(advisory_tab)
    advisory_frame.pack(pady=10, fill='both', expand=True)

    # Create Treeview for advisory data
    advisory_tree = ttk.Treeview(advisory_frame)
    advisory_tree["columns"] = ("City", "Avg AQI", "Max AQI", "Max Date", "Health Risk", "Recommendations")

    # Define columns
    advisory_tree.column("#0", width=0, stretch=tk.NO)
    advisory_tree.column("City", width=100, minwidth=100)
    advisory_tree.column("Avg AQI", width=70, minwidth=70)
    advisory_tree.column("Max AQI", width=70, minwidth=70)
    advisory_tree.column("Max Date", width=100, minwidth=100)
    advisory_tree.column("Health Risk", width=150, minwidth=150)
    advisory_tree.column("Recommendations", width=300, minwidth=300)

    advisory_tree.heading("#0", text="", anchor=tk.W)
    advisory_tree.heading("City", text="City", anchor=tk.W)
    advisory_tree.heading("Avg AQI", text="Avg AQI", anchor=tk.W)
    advisory_tree.heading("Max AQI", text="Max AQI", anchor=tk.W)
    advisory_tree.heading("Max Date", text="Max Date", anchor=tk.W)
    advisory_tree.heading("Health Risk", text="Health Risk", anchor=tk.W)
    advisory_tree.heading("Recommendations", text="Recommendations", anchor=tk.W)

    # Add a scrollbar
    advisory_scrollbar = ttk.Scrollbar(advisory_frame, orient="vertical", command=advisory_tree.yview)
    advisory_tree.configure(yscrollcommand=advisory_scrollbar.set)

    # Pack the scrollbar and tree
    advisory_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    advisory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Insert data
    for i, row in advisory_df.iterrows():
        advisory_tree.insert(
            "",
            tk.END,
            values=(
                row['city'],
                row['average_aqi'],
                row['max_aqi'],
                row['max_aqi_date'],
                row['worst_health_risk'].split(' - ')[0],
                row['recommendations']
            )
        )

    # Map tab content
    map_frame = ttk.Frame(map_tab)
    map_frame.pack(pady=10, fill='both', expand=True)

    # Add date selector for map
    map_date_frame = ttk.Frame(map_frame)
    map_date_frame.pack(fill=tk.X, pady=10)

    map_date_label = ttk.Label(map_date_frame, text="Select forecast date:")
    map_date_label.grid(row=0, column=0, padx=5, pady=5)

    # Get unique dates from predictions for dropdown
    unique_dates = sorted(predictions_df['datetime'].dt.date.unique())
    date_strings = [d.strftime('%Y-%m-%d') for d in unique_dates]

    map_date_var = tk.StringVar()
    map_date_dropdown = ttk.Combobox(map_date_frame, textvariable=map_date_var, values=date_strings)
    map_date_dropdown.grid(row=0, column=1, padx=5, pady=5)
    map_date_dropdown.current(0)

    # Generate and display map button
    def show_map():
        selected_date_str = map_date_var.get()
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
        map_path = create_map(df, predictions_df, selected_date)
        webbrowser.open('file://' + os.path.realpath(map_path))

    map_button = ttk.Button(map_frame, text="Generate and Show Map", command=show_map)
    map_button.pack(pady=10)

    # Map preview label
    map_preview_label = ttk.Label(map_frame,
                                  text="Click the button above to generate and open the interactive map in your browser.")
    map_preview_label.pack(pady=20)

    # Heatmap tab content
    heatmap_frame = ttk.Frame(heatmap_tab)
    heatmap_frame.pack(pady=10, fill='both', expand=True)

    # Add date selector for heatmap
    heatmap_date_frame = ttk.Frame(heatmap_frame)
    heatmap_date_frame.pack(fill=tk.X, pady=10)

    heatmap_date_label = ttk.Label(heatmap_date_frame, text="Select forecast date:")
    heatmap_date_label.grid(row=0, column=0, padx=5, pady=5)

    heatmap_date_var = tk.StringVar()
    heatmap_date_dropdown = ttk.Combobox(heatmap_date_frame, textvariable=heatmap_date_var, values=date_strings)
    heatmap_date_dropdown.grid(row=0, column=1, padx=5, pady=5)
    heatmap_date_dropdown.current(0)

    # Generate and display heatmap button
    def show_heatmap():
        selected_date_str = heatmap_date_var.get()
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
        heatmap_path = create_heatmap(df, predictions_df, selected_date)
        webbrowser.open('file://' + os.path.realpath(heatmap_path))

    heatmap_button = ttk.Button(heatmap_frame, text="Generate and Show Heatmap", command=show_heatmap)
    heatmap_button.pack(pady=10)

    # Heatmap preview label
    heatmap_preview_label = ttk.Label(heatmap_frame,
                                      text="Click the button above to generate and open the interactive heatmap in your browser.")
    heatmap_preview_label.pack(pady=20)

    spatial_frame = ttk.Frame(spatial_tab)
    spatial_frame.pack(pady=10, fill='both', expand=True)

    # Create a title frame at the top
    spatial_title_frame = ttk.Frame(spatial_frame)
    spatial_title_frame.pack(side=tk.TOP, fill=tk.X)

    # Add title label with custom styling
    spatial_title_label = ttk.Label(
        spatial_title_frame,
        text="Air Quality Spatial Analysis",
        font=("Arial", 14, "bold")
    )
    spatial_title_label.pack(pady=5)

    # Create a frame for the plot
    spatial_plot_frame = ttk.Frame(spatial_frame)
    spatial_plot_frame.pack(side=tk.LEFT, fill='both', expand=True)

    # Create a frame for the recommendations/legend
    spatial_info_frame = ttk.Frame(spatial_frame)
    spatial_info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    # Create figure for spatial analysis
    spatial_fig = Figure(figsize=(8, 5))
    spatial_ax = spatial_fig.add_subplot(111)

    # Create canvas for figure
    spatial_canvas = FigureCanvasTkAgg(spatial_fig, master=spatial_plot_frame)
    spatial_canvas_widget = spatial_canvas.get_tk_widget()
    spatial_canvas_widget.pack(fill='both', expand=True)

    # Add analysis controls frame at the bottom
    spatial_controls_frame = ttk.Frame(spatial_plot_frame)
    spatial_controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    # Add analysis type selector
    analysis_frame = ttk.Frame(spatial_controls_frame)
    analysis_frame.pack(fill=tk.X, pady=5)

    analysis_label = ttk.Label(analysis_frame, text="Select analysis type:")
    analysis_label.grid(row=0, column=0, padx=5, pady=5)

    analysis_types = ["Regional AQI Comparison", "Pollution Component Analysis", "Temporal Patterns"]
    analysis_var = tk.StringVar()
    analysis_dropdown = ttk.Combobox(analysis_frame, textvariable=analysis_var, values=analysis_types)
    analysis_dropdown.grid(row=0, column=1, padx=5, pady=5)
    analysis_dropdown.current(0)

    # Add city range control similar to health risk tab
    spatial_city_start_index = 0
    spatial_max_cities_per_view = 15  # Show this many cities at once

    # Add scroll control frame
    scroll_frame = ttk.Frame(spatial_controls_frame)
    scroll_frame.pack(fill=tk.X, side=tk.BOTTOM)

    # Add scrollbar label
    scrollbar_label = ttk.Label(scroll_frame, text="Scroll cities:")
    scrollbar_label.pack(side=tk.LEFT, padx=5)

    # Add scroll status
    spatial_scroll_status = ttk.Label(scroll_frame, text="", width=25)
    spatial_scroll_status.pack(side=tk.LEFT, padx=5)

    # Define scroll functions
    def spatial_scroll_prev():
        nonlocal spatial_city_start_index
        if spatial_city_start_index >= spatial_max_cities_per_view:
            spatial_city_start_index -= spatial_max_cities_per_view
        else:
            spatial_city_start_index = 0
        update_spatial_analysis()

    def spatial_scroll_next():
        nonlocal spatial_city_start_index
        city_count = len(df['city_name'].unique())
        if spatial_city_start_index + spatial_max_cities_per_view < city_count:
            spatial_city_start_index += spatial_max_cities_per_view
            update_spatial_analysis()

    # Update spatial analysis plot
    def update_spatial_analysis():
        nonlocal spatial_city_start_index
        city_count = len(df['city_name'].unique())
        if spatial_city_start_index + spatial_max_cities_per_view < city_count:
            spatial_city_start_index += spatial_max_cities_per_view
            update_spatial_analysis()

    # Add scroll buttons
    spatial_prev_scroll_btn = ttk.Button(scroll_frame, text="◀ Previous", command=spatial_scroll_prev)
    spatial_prev_scroll_btn.pack(side=tk.LEFT, padx=5)

    spatial_next_scroll_btn = ttk.Button(scroll_frame, text="Next ▶", command=spatial_scroll_next)
    spatial_next_scroll_btn.pack(side=tk.LEFT, padx=5)

    # Add button to update analysis
    analysis_button = ttk.Button(analysis_frame, text="Update Analysis", command=lambda: update_spatial_analysis())
    analysis_button.grid(row=0, column=2, padx=10, pady=5)

    # Recommendations text widget
    recommendations_title = ttk.Label(spatial_info_frame, text="Recommendations", font=("Arial", 12, "bold"))
    recommendations_title.pack(pady=(0, 5))

    recommendations_text = tk.Text(spatial_info_frame, wrap=tk.WORD, width=30, height=20)
    recommendations_text.pack(fill=tk.BOTH, expand=True)

    # Update spatial analysis plot
    def update_spatial_analysis():
        spatial_ax.clear()
        analysis_type = analysis_var.get()

        # Get all cities
        all_cities = sorted(df['city_name'].unique())
        city_count = len(all_cities)

        # Handle city range for display
        if city_count > spatial_max_cities_per_view:
            end_index = min(spatial_city_start_index + spatial_max_cities_per_view, city_count)
            visible_cities = all_cities[spatial_city_start_index:end_index]
            spatial_scroll_status.config(
                text=f"Showing {spatial_city_start_index + 1}-{end_index} of {city_count} cities")
        else:
            visible_cities = all_cities
            spatial_scroll_status.config(text=f"Showing all {city_count} cities")

        if analysis_type == "Regional AQI Comparison":
            # Average AQI by city with color-coded bars based on health risk
            city_data = df[df['city_name'].isin(visible_cities)]
            city_avg = city_data.groupby('city_name')['main.aqi'].mean().sort_values(ascending=False)

            # Create colors based on AQI values
            colors = [get_aqi_color(aqi) for aqi in city_avg.values]

            # Plot bars with custom colors (horizontal bars for better readability with many cities)
            bars = spatial_ax.barh(city_avg.index, city_avg.values, color=colors)

            spatial_ax.set_title('Average AQI by City')
            spatial_ax.set_xlabel('Average AQI')
            spatial_ax.set_ylabel('City')

            # Add grid for readability
            spatial_ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # Update recommendations
            recommendation_text = (
                "Recommendations by AQI Level:\n\n"
                "Good (0-50): No significant health risks. Enjoy outdoor activities as usual.\n\n"
                "Moderate (51-100): Sensitive individuals (e.g., those with asthma, allergies, or heart conditions) should consider wearing a mask outdoors. Limit prolonged outdoor exertion if you experience discomfort.\n\n"
                "Unhealthy for Sensitive Groups (101-150): Children, elderly, and individuals with respiratory or heart conditions should wear masks outdoors. Avoid prolonged outdoor activities; opt for indoor exercises instead.\n\n"
                "Unhealthy (151-200): Everyone should wear a mask when outside. Avoid prolonged outdoor exertion; reschedule outdoor activities for days with better air quality. Keep windows and doors closed to prevent polluted air from entering your home.\n\n"
                "Very Unhealthy (201-300): Everyone should avoid outdoor activities. Wear a high-quality mask (e.g., N95 or KN95) if you must go outside. Use air purifiers indoors and keep windows closed. Stay hydrated and monitor for symptoms like coughing or shortness of breath.\n\n"
                "Hazardous (301+): EMERGENCY CONDITIONS: Stay indoors as much as possible. Wear a high-quality mask (e.g., N95 or KN95) if you must go outside. Use air purifiers and keep windows and doors tightly closed. Avoid physical exertion, even indoors. Seek medical advice if you experience severe symptoms like difficulty breathing or chest pain."
            )

        elif analysis_type == "Pollution Component Analysis":
            # Select pollution components
            components = ['components.pm2_5', 'components.pm10', 'components.o3', 'components.no2', 'components.so2']
            component_labels = ['PM2.5', 'PM10', 'Ozone', 'NO2', 'SO2']

            # Filter for visible cities
            component_data = df[df['city_name'].isin(visible_cities)].groupby('city_name')[components].mean()

            # Plot stacked bar chart (horizontal for better readability)
            component_data.plot(kind='barh', stacked=True, ax=spatial_ax, colormap='tab10')

            # Set labels
            spatial_ax.set_title('Average Pollution Components by City')
            spatial_ax.set_xlabel('Concentration')
            spatial_ax.set_ylabel('City')
            spatial_ax.legend(component_labels, loc='lower right')

            # Add grid for readability
            spatial_ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # Update recommendations
            recommendation_text = "Health Impacts by Pollutant:\n\n" + \
                                  "PM2.5: Can penetrate deep into lungs, wear N95 masks when levels are high\n\n" + \
                                  "PM10: Can irritate respiratory system, stay indoors during dust events\n\n" + \
                                  "Ozone: Can trigger asthma, avoid outdoor exercise in afternoons\n\n" + \
                                  "NO2: Can irritate airways, limit time near heavy traffic\n\n" + \
                                  "SO2: Can harm respiratory system, be cautious near industrial areas"

        elif analysis_type == "Temporal Patterns":
            # For temporal patterns, we don't need city filtering as we're looking at overall patterns

            # Hourly patterns across all cities
            hourly_avg = df.groupby('hour')['main.aqi'].mean()

            # Color regions by time of day
            morning = spatial_ax.axvspan(5, 11, alpha=0.2, color='yellow', label='Morning')
            afternoon = spatial_ax.axvspan(11, 17, alpha=0.2, color='orange', label='Afternoon')
            evening = spatial_ax.axvspan(17, 22, alpha=0.2, color='blue', label='Evening')
            night = spatial_ax.axvspan(22, 24, alpha=0.2, color='navy', label='Night')
            night2 = spatial_ax.axvspan(0, 5, alpha=0.2, color='navy')

            # Plot line
            spatial_ax.plot(hourly_avg.index, hourly_avg.values, 'b-o', linewidth=2)

            # Set labels
            spatial_ax.set_title('Average AQI by Hour of Day')
            spatial_ax.set_xlabel('Hour of Day')
            spatial_ax.set_ylabel('Average AQI')
            spatial_ax.set_xticks(range(0, 24, 2))
            spatial_ax.grid(True, linestyle='--', alpha=0.7)
            spatial_ax.legend(loc='upper right')

            # Update recommendations
            recommendation_text = "Time-Based Recommendations:\n\n" + \
                                  "Morning (5-11 AM): Good time for outdoor activities when AQI is typically lower\n\n" + \
                                  "Afternoon (11 AM-5 PM): Higher pollution levels, limit strenuous outdoor activities\n\n" + \
                                  "Evening (5-10 PM): Check local AQI before outdoor dining or exercise\n\n" + \
                                  "Night (10 PM-5 AM): Keep windows closed if you live near industrial areas or major roads"

        # Update the recommendations text
        recommendations_text.config(state=tk.NORMAL)
        recommendations_text.delete(1.0, tk.END)
        recommendations_text.insert(tk.END, recommendation_text)
        recommendations_text.config(state=tk.DISABLED)

        # Adjust layout
        spatial_fig.tight_layout()
        spatial_canvas.draw()

    # Add button to update analysis
    analysis_button = ttk.Button(analysis_frame, text="Update Analysis", command=update_spatial_analysis)
    analysis_button.grid(row=0, column=2, padx=10, pady=5)

    # Set up event bindings
    city_dropdown.bind("<<ComboboxSelected>>", lambda event: update_city_plot())
    analysis_dropdown.bind("<<ComboboxSelected>>", lambda event: update_spatial_analysis())

    # Initial plot updates
    update_city_plot()
    update_risk_plot()
    update_spatial_analysis()

    # Start the mainloop
    root.mainloop()


# Main application flow
if __name__ == "__main__":
    # Check if data file exists
    file_path = "updated_air_quality1.csv"
    if os.path.exists(file_path):
        df = load_and_preprocess_data(file_path)
    else:
        # Generate synthetic data for demonstration
        print("Generating synthetic data for demonstration...")
        df = generate_synthetic_data(pd.DataFrame())
        df.to_csv(file_path, index=False)
        print(f"Synthetic data saved to {file_path}")

    print("Training prediction model...")
    model, scaler, features = train_prediction_model(df)

    print("Generating predictions...")
    predictions_df = predict_future_aqi(model, scaler, features, df, future_days=7)

    print("Launching application...")
    run_application(df, predictions_df)