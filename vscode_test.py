import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# API configuration
API_KEY = "647626c3e259fcc8e1b5e8f16fa5dad4"  # Replace with your OpenWeatherMap API key
CITY = "London"  # Replace with your city
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

def get_weather_forecast():
    params = {
        'q': CITY,
        'appid': API_KEY,
        'units': 'metric'  # For Celsius
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def process_forecast(data):
    if not data:
        return None
    
    forecast_list = []
    for item in data['list'][:7]:  # Get next 7 days
        date = datetime.fromtimestamp(item['dt'])
        forecast_list.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature': round(item['main']['temp'], 1),
            'Weather': item['weather'][0]['main'],
            'Description': item['weather'][0]['description']
        })
    
    return pd.DataFrame(forecast_list)

def main():
    print(f"Fetching 7-day weather forecast for {CITY}...")
    weather_data = get_weather_forecast()
    
    if weather_data:
        forecast_df = process_forecast(weather_data)
        if forecast_df is not None:
            print("\n7-Day Weather Forecast:")
            print(forecast_df.to_string(index=False))
        else:
            print("Failed to process forecast data.")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()
