import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import geopandas as gpd
import io
import requests
from geopy.geocoders import Nominatim
import os
import importlib.util
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Rainfall Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data(pincode_df=None, file_path='2000_2022.nc'):
    try:
        # Check if netCDF4 is installed
        if importlib.util.find_spec('netCDF4') is None:
            st.error("The 'netCDF4' package is not installed. Please install it using: pip install netCDF4")
            return None
        
        # Check if dask is installed for chunking
        use_chunks = importlib.util.find_spec('dask') is not None
        if not use_chunks:
            st.warning("The 'dask' package is not installed. Loading without chunking, which may use more memory.")
        
        # Google Drive URL for subsetted dataset
        url = "https://drive.google.com/uc?export=download&id=1uvAiLbh1j-xykSuk-Mc0jM24hREDG1Ti"
        
        if not os.path.exists(file_path):
            st.info(f"Downloading dataset from Google Drive (~1–2 GB, may take a few minutes)...")
            retries = 3
            for attempt in range(retries):
                try:
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success(f"Dataset downloaded successfully to {file_path}")
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 403:
                        st.error(
                            f"HTTP 403 Forbidden: Cannot download dataset. "
                            "Please ensure the Google Drive file (https://drive.google.com/file/d/1uvAiLbh1j-xykSuk-Mc0jM24hREDG1Ti/view?usp=drive_link) "
                            "is shared as 'Anyone with the link' (Viewer). "
                            "Right-click the file in Google Drive, select 'Share', set 'General access' to 'Anyone with the link', and retry."
                        )
                    elif response.status_code == 404:
                        st.error(
                            f"HTTP 404 Not Found: File not found on Google Drive. "
                            "Please verify the file ID (1uvAiLbh1j-xykSuk-Mc0jM24hREDG1Ti) is correct or re-upload "
                            "Indian_Daily_Rainfall_2000_2022.nc to Google Drive and update the URL in app.py."
                        )
                    else:
                        st.error(f"HTTP Error {response.status_code}: {e}")
                    if attempt == retries - 1:
                        st.error("All download attempts failed. Please check the Google Drive link and sharing settings.")
                        return None
                    time.sleep(2)
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error on attempt {attempt + 1}/{retries}: {e}. Retrying...")
                    if attempt == retries - 1:
                        st.error(
                            "Failed to download dataset after retries. "
                            "Please check your network connection and ensure the Google Drive file is accessible."
                        )
                        return None
                    time.sleep(2)
        
        # Open dataset, use chunking if dask is available
        open_kwargs = {'engine': 'netcdf4'}
        if use_chunks:
            open_kwargs['chunks'] = {'TIME': 100}
        
        ds = xr.open_dataset(file_path, **open_kwargs)
        
        # Subset for pin code if provided
        if pincode_df is not None:
            lat = pincode_df['latitude'].iloc[0]
            lon = pincode_df['longitude'].iloc[0]
            # Validate coordinates within India's approximate bounds
            if not (6 <= lat <= 36 and 66 <= lon <= 100):
                st.error(f"Pin code coordinates ({lat}, {lon}) are outside India's spatial extent (~6–36°N, 66–100°E).")
                return None
            # Store original bounds for validation
            lat_bounds = (float(ds.LATITUDE.min()), float(ds.LATITUDE.max()))
            lon_bounds = (float(ds.LONGITUDE.min()), float(ds.LONGITUDE.max()))
            if not (lat_bounds[0] <= lat <= lat_bounds[1] and lon_bounds[0] <= lon <= lon_bounds[1]):
                st.warning(f"Pin code coordinates ({lat}, {lon}) are outside dataset bounds ({lat_bounds}, {lon_bounds}). Using nearest grid point.")
            # Select nearest grid point
            ds = ds.sel(LATITUDE=lat, LONGITUDE=lon, method='nearest')
        
        return ds
    except ValueError as e:
        st.error(f"Failed to open dataset: {e}. Ensure the file is a valid NetCDF and 'netCDF4' is installed.")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None

# Fetch pincode data and coordinates
def fetch_pincode_data(pincode, retries=3):
    for attempt in range(retries):
        try:
            url = f"https://api.postalpincode.in/pincode/{pincode}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data[0]['Status'] == 'Success' and data[0]['PostOffice']:
                post_office = data[0]['PostOffice'][0]
                district = post_office.get('District', '')
                state = post_office.get('State', '')
                geolocator = Nominatim(user_agent="rainfall_dashboard")
                query = f"{pincode}, {district}, {state}, India"
                location = geolocator.geocode(query, timeout=10)
                if location:
                    return pd.DataFrame({
                        'pincode': [pincode],
                        'latitude': [location.latitude],
                        'longitude': [location.longitude],
                        'district': [district],
                        'state': [state]
                    })
                st.error(f"Coordinates not found for pin code {pincode}.")
                return None
            st.error(f"Invalid pin code: {pincode}")
            return None
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Failed to fetch pin code data after {retries} attempts: {e}")
                return None
            time.sleep(1)

# Convert pincode data to GeoDataFrame
def create_pincode_gdf(pincode_df):
    if pincode_df is not None:
        return gpd.GeoDataFrame(
            pincode_df,
            geometry=gpd.points_from_xy(pincode_df.longitude, pincode_df.latitude),
            crs="EPSG:4326"
        )
    return None

# Get nearest rainfall value
def get_nearest_rainfall(_ds, selected_date):
    try:
        rainfall = _ds.RAINFALL.sel(TIME=selected_date).values
        if np.isnan(rainfall):
            st.warning("No rainfall data available for the selected date.")
            return np.nan
        return float(rainfall)
    except Exception as e:
        st.warning(f"Failed to retrieve rainfall data: {e}")
        return np.nan

# Cache trend calculations
@st.cache_data
def calculate_trends(_ds, start_year=None, end_year=None, _gdf=None):
    try:
        if _gdf is not None:
            yearly_means = _ds.RAINFALL.groupby('TIME.year').mean()
            years = yearly_means.year.values
            y = yearly_means.values
        else:
            yearly_means = _ds.RAINFALL.groupby('TIME.year').mean()
            years = yearly_means.year.values
            y = np.nanmean(yearly_means.values, axis=(1, 2))
        if start_year and end_year:
            mask = (years >= start_year) & (years <= end_year)
            years = years[mask]
            y = y[mask]
        X = years.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        trend = reg.coef_[0]
        return trend, years, y
    except Exception as e:
        st.error(f"Trend calculation failed: {e}")
        return None, None, None

# Cache cumulative calculations
@st.cache_data
def calculate_cumulative_averages(_ds, period, _gdf=None):
    try:
        if _gdf is not None:
            if period == 'monthly':
                monthly_sum = _ds.RAINFALL.groupby('TIME.month').sum()
                years_count = len(np.unique(_ds.TIME.dt.year.values))
                cumulative_avg = monthly_sum.values / years_count if years_count > 0 else monthly_sum.values
                labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                return cumulative_avg, labels, np.arange(1, 13)
            elif period == 'weekly':
                weekly_sum = _ds.RAINFALL.groupby('TIME.week').sum()
                years_count = len(np.unique(_ds.TIME.dt.year.values))
                cumulative_avg = weekly_sum.values / years_count if years_count > 0 else weekly_sum.values
                labels = [f"Week {i}" for i in range(1, len(weekly_sum.week) + 1)]
                return cumulative_avg, labels, weekly_sum.week.values
            elif period == 'yearly':
                yearly_sum = _ds.RAINFALL.groupby('TIME.year').sum()
                cumulative_avg = yearly_sum.values
                labels = yearly_sum.year.values
                return cumulative_avg, labels, labels
        else:
            if period == 'monthly':
                monthly_sum = _ds.RAINFALL.groupby('TIME.month').sum()
                years_count = len(np.unique(_ds.TIME.dt.year.values))
                cumulative_avg = np.nanmean(monthly_sum.values, axis=(1, 2)) / years_count
                labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                return cumulative_avg, labels, np.arange(1, 13)
            elif period == 'weekly':
                weekly_sum = _ds.RAINFALL.groupby('TIME.week').sum()
                years_count = len(np.unique(_ds.TIME.dt.year.values))
                cumulative_avg = np.nanmean(weekly_sum.values, axis=(1, 2)) / years_count
                labels = [f"Week {i}" for i in range(1, len(weekly_sum.week) + 1)]
                return cumulative_avg, labels, weekly_sum.week.values
            elif period == 'yearly':
                yearly_sum = _ds.RAINFALL.groupby('TIME.year').sum()
                cumulative_avg = np.nanmean(yearly_sum.values, axis=(1, 2))
                labels = yearly_sum.year.values
                return cumulative_avg, labels, labels
    except Exception as e:
        st.error(f"Cumulative calculation failed: {e}")
        return None, None, None

# Forecast rainfall
def forecast_rainfall(yearly_data, years, steps=5):
    try:
        model = ARIMA(yearly_data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast, np.arange(years[-1] + 1, years[-1] + steps + 1)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")
        return None, None

# Create map
def create_map(_ds, selected_date, _gdf=None):
    try:
        if _gdf is not None and _gdf.geometry.type.str.contains('Point').all():
            center = [_gdf.geometry.y.iloc[0], _gdf.geometry.x.iloc[0]]
            m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
            rainfall_value = get_nearest_rainfall(_ds, pd.to_datetime(selected_date))
            popup_text = f"Pin Code: {_gdf['pincode'].iloc[0]}<br>Rainfall: {rainfall_value:.2f} mm" if not np.isnan(rainfall_value) else "No data"
            folium.Marker(
                location=center,
                popup=popup_text,
                icon=folium.Icon(color='blue')
            ).add_to(m)
        else:
            if _ds.LATITUDE.ndim == 1 and _ds.LONGITUDE.ndim == 1:
                center = [float(_ds.LATITUDE.mean()), float(_ds.LONGITUDE.mean())]
                m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
                rainfall_data = _ds.RAINFALL.sel(TIME=selected_date).values
                lat = _ds.LATITUDE.values
                lon = _ds.LONGITUDE.values
                heat_data = []
                for i in range(len(lat)):
                    for j in range(len(lon)):
                        if not np.isnan(rainfall_data[i, j]):
                            heat_data.append([float(lat[i]), float(lon[j]), float(rainfall_data[i, j])])
                HeatMap(heat_data, radius=15, blur=20).add_to(m)
            else:
                st.error("Cannot generate heatmap for subsetted dataset (single grid point).")
                return None
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Map creation failed: {e}")
        return None

# Main dashboard
def main():
    st.title("🌧️ Rainfall Analysis Dashboard for Pin Code")
    st.markdown("Analyze rainfall patterns (2000–2022) using the nearest grid point for a specific pin code.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.sidebar.header("⚙️ Analysis Controls")
        pincode_input = st.sidebar.text_input(
            "Enter Pin Code (e.g., 110001)",
            help="Enter a single pin code for analysis"
        )
        gdf = None
        pincode_df = None
        if pincode_input:
            with st.spinner("Fetching pin code data..."):
                pincode_df = fetch_pincode_data(pincode_input)
                gdf = create_pincode_gdf(pincode_df)
        
        ds = None
        if pincode_df is not None:
            ds = load_data(pincode_df)
        elif pincode_input:
            st.error("Cannot load data without valid pin code coordinates.")
            return
        else:
            ds = load_data()
        
        if ds is None:
            st.error("Failed to load dataset. Please check logs or try again.")
            return
        
        min_date = pd.to_datetime(ds.TIME.values.min())
        max_date = pd.to_datetime(ds.TIME.values.max())
        selected_date = st.sidebar.date_input(
            "Select Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            [
                "Daily Map",
                "Monthly Average",
                "Yearly Trends",
                "Seasonal Patterns",
                "Forecast",
                "Monthly Cumulative Average",
                "Weekly Cumulative Average",
                "Yearly Cumulative Average",
                "Pin Code Rainfall"
            ]
        )
        
        if analysis_type in ["Yearly Trends", "Forecast"]:
            start_year = st.sidebar.slider(
                "Start Year",
                int(ds.TIME.dt.year.min()),
                int(ds.TIME.dt.year.max()),
                int(ds.TIME.dt.year.min())
            )
            end_year = st.sidebar.slider(
                "End Year",
                start_year,
                int(ds.TIME.dt.year.max()),
                int(ds.TIME.dt.year.max())
            )
        else:
            start_year, end_year = None, None
    
    with col2:
        st.subheader(f"{analysis_type} Visualization")
        if gdf is None and analysis_type != "Daily Map":
            st.error("Please enter a valid pin code.")
            return
        
        if analysis_type == "Daily Map":
            m = create_map(ds, pd.to_datetime(selected_date), gdf)
            if m:
                folium_static(m, width=800, height=500)
            if gdf is not None:
                rainfall_value = get_nearest_rainfall(ds, pd.to_datetime(selected_date))
                if not np.isnan(rainfall_value):
                    st.metric("Rainfall at Pin Code", f"{rainfall_value:.2f} mm")
                else:
                    st.metric("Rainfall at Pin Code", "No data")
        
        elif analysis_type == "Monthly Average":
            monthly_avg = []
            for month in range(1, 13):
                monthly_data = ds.RAINFALL.where(ds.TIME.dt.month == month).mean('TIME')
                monthly_avg.append(float(monthly_data) if not np.isnan(monthly_data) else np.nan)
            fig = px.line(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_avg,
                labels={'x': 'Month', 'y': 'Average Rainfall (mm)'},
                title=f'Monthly Average Rainfall at Pin Code {pincode_input} (2000-2022)',
                template='plotly_white'
            )
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Yearly Trends":
            trend, years, rainfall = calculate_trends(ds, start_year, end_year, gdf)
            if years is not None:
                fig = px.scatter(
                    x=years,
                    y=rainfall,
                    trendline="ols",
                    labels={'x': 'Year', 'y': 'Average Rainfall (mm)'},
                    title=f'Yearly Rainfall Trend at Pin Code {pincode_input} (Slope: {trend:.4f} mm/year)',
                    template='plotly_white'
                )
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Seasonal Patterns":
            seasonal_avg = []
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            for season in seasons:
                seasonal_data = ds.RAINFALL.where(ds.TIME.dt.season == season).mean('TIME')
                seasonal_avg.append(float(seasonal_data) if not np.isnan(seasonal_data) else np.nan)
            fig = px.bar(
                x=seasons,
                y=seasonal_avg,
                labels={'x': 'Season', 'y': 'Average Rainfall (mm)'},
                title=f'Seasonal Rainfall Distribution at Pin Code {pincode_input}',
                template='plotly_white',
                color=seasons
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Forecast":
            trend, years, rainfall = calculate_trends(ds, start_year, end_year, gdf)
            if years is not None:
                forecast, forecast_years = forecast_rainfall(rainfall, years)
                if forecast is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=rainfall,
                        mode='lines+markers',
                        name='Historical'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_years,
                        y=forecast,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(dash='dash')
                    ))
                    fig.update_layout(
                        title=f'Rainfall Forecast at Pin Code {pincode_input}',
                        xaxis_title='Year',
                        yaxis_title='Average Rainfall (mm)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Unable to generate forecast.")
        
        elif analysis_type == "Monthly Cumulative Average":
            cumulative_avg, labels, x_values = calculate_cumulative_averages(ds, 'monthly', gdf)
            if cumulative_avg is not None:
                fig = px.line(
                    x=labels,
                    y=cumulative_avg,
                    labels={'x': 'Month', 'y': 'Cumulative Average Rainfall (mm)'},
                    title=f'Monthly Cumulative Average Rainfall at Pin Code {pincode_input} (2000-2022)',
                    template='plotly_white'
                )
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(fig, use_container_width=True)
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Mean Cumulative Rainfall", f"{float(cumulative_avg.mean()):.2f} mm")
                stats_col2.metric("Max Cumulative Rainfall", f"{float(cumulative_avg.max()):.2f} mm")
                stats_col3.metric("Min Cumulative Rainfall", f"{float(cumulative_avg.min()):.2f} mm")
        
        elif analysis_type == "Weekly Cumulative Average":
            cumulative_avg, labels, x_values = calculate_cumulative_averages(ds, 'weekly', gdf)
            if cumulative_avg is not None:
                fig = px.line(
                    x=x_values,
                    y=cumulative_avg,
                    labels={'x': 'Week', 'y': 'Cumulative Average Rainfall (mm)'},
                    title=f'Weekly Cumulative Average Rainfall at Pin Code {pincode_input} (2000-2022)',
                    template='plotly_white'
                )
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(fig, use_container_width=True)
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Mean Cumulative Rainfall", f"{float(cumulative_avg.mean()):.2f} mm")
                stats_col2.metric("Max Cumulative Rainfall", f"{float(cumulative_avg.max()):.2f} mm")
                stats_col3.metric("Min Cumulative Rainfall", f"{float(cumulative_avg.min()):.2f} mm")
        
        elif analysis_type == "Yearly Cumulative Average":
            cumulative_avg, labels, x_values = calculate_cumulative_averages(ds, 'yearly', gdf)
            if cumulative_avg is not None:
                fig = px.bar(
                    x=x_values,
                    y=cumulative_avg,
                    labels={'x': 'Year', 'y': 'Cumulative Rainfall (mm)'},
                    title=f'Yearly Cumulative Rainfall at Pin Code {pincode_input} (2000-2022)',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Mean Cumulative Rainfall", f"{float(cumulative_avg.mean()):.2f} mm")
                stats_col2.metric("Max Cumulative Rainfall", f"{float(cumulative_avg.max()):.2f} mm")
                stats_col3.metric("Min Cumulative Rainfall", f"{float(cumulative_avg.min()):.2f} mm")
        
        elif analysis_type == "Pin Code Rainfall":
            rainfall_value = get_nearest_rainfall(ds, pd.to_datetime(selected_date))
            st.subheader(f"Rainfall for Pin Code {pincode_input}")
            if not np.isnan(rainfall_value):
                st.metric("Rainfall", f"{rainfall_value:.2f} mm")
            else:
                st.metric("Rainfall", "No data")
            m = create_map(ds, pd.to_datetime(selected_date), gdf)
            if m:
                folium_static(m, width=800, height=500)
    
    # Data export
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Data")
    export_format = st.sidebar.selectbox("Export Format", ["CSV"])
    if st.sidebar.button("Download Data"):
        if gdf is None:
            st.error("Please enter a valid pin code.")
        else:
            try:
                if analysis_type == "Daily Map" or analysis_type == "Pin Code Rainfall":
                    rainfall_value = get_nearest_rainfall(ds, pd.to_datetime(selected_date))
                    data = pd.DataFrame({
                        'Pincode': [pincode_input],
                        'Date': [selected_date],
                        'Rainfall': [rainfall_value]
                    })
                elif analysis_type == "Monthly Average":
                    monthly_avg = []
                    for month in range(1, 13):
                        monthly_data = ds.RAINFALL.where(ds.TIME.dt.month == month).mean('TIME')
                        monthly_avg.append(float(monthly_data) if not np.isnan(monthly_data) else np.nan)
                    data = pd.DataFrame({
                        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        'Average_Rainfall': monthly_avg
                    })
                elif analysis_type == "Yearly Trends":
                    trend, years, rainfall = calculate_trends(ds, start_year, end_year, gdf)
                    if years is not None:
                        data = pd.DataFrame({'Year': years, 'Rainfall': rainfall})
                    else:
                        st.error("No data to export.")
                        return
                elif analysis_type == "Seasonal Patterns":
                    seasonal_avg = []
                    seasons = ['DJF', 'MAM', 'JJA', 'SON']
                    for season in seasons:
                        seasonal_data = ds.RAINFALL.where(ds.TIME.dt.season == season).mean('TIME')
                        seasonal_avg.append(float(seasonal_data) if not np.isnan(seasonal_data) else np.nan)
                    data = pd.DataFrame({'Season': seasons, 'Average_Rainfall': seasonal_avg})
                elif analysis_type == "Forecast":
                    trend, years, rainfall = calculate_trends(ds, start_year, end_year, gdf)
                    if years is not None:
                        forecast, forecast_years = forecast_rainfall(rainfall, years)
                        data = pd.DataFrame({
                            'Year': np.concatenate([years, forecast_years]),
                            'Rainfall': np.concatenate([rainfall, forecast]) if forecast is not None else rainfall
                        })
                    else:
                        st.error("No data to export.")
                        return
                elif analysis_type == "Monthly Cumulative Average":
                    cumulative_avg, _, _ = calculate_cumulative_averages(ds, 'monthly', gdf)
                    if cumulative_avg is not None:
                        data = pd.DataFrame({
                            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            'Cumulative_Average_Rainfall': cumulative_avg
                        })
                    else:
                        st.error("No data to export.")
                        return
                elif analysis_type == "Weekly Cumulative Average":
                    cumulative_avg, _, x_values = calculate_cumulative_averages(ds, 'weekly', gdf)
                    if cumulative_avg is not None:
                        data = pd.DataFrame({
                            'Week': x_values,
                            'Cumulative_Average_Rainfall': cumulative_avg
                        })
                    else:
                        st.error("No data to export.")
                        return
                elif analysis_type == "Yearly Cumulative Average":
                    cumulative_avg, years, _ = calculate_cumulative_averages(ds, 'yearly', gdf)
                    if cumulative_avg is not None:
                        data = pd.DataFrame({
                            'Year': years,
                            'Cumulative_Rainfall': cumulative_avg
                        })
                    else:
                        st.error("No data to export.")
                        return
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=buffer.getvalue(),
                    file_name=f"{analysis_type.lower().replace(' ', '_')}_{pincode_input}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

if __name__ == "__main__":
    main()
