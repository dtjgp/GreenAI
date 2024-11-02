import torch
from torchvision.models import resnet50, resnet18
from googlenet_FashionMnist import Googlenet
import requests
import json
from datetime import datetime

def get_weather(api_key, city):
    """
    Fetch weather data for a given city using OpenWeatherMap API
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # for Celsius
    }
    
    try:
        response = requests.get(base_url, params=params)
        weather_data = response.json()
        if response.status_code == 401:
            return "Error: Invalid API key. Please wait 1-2 hours for a new key to activate."
        elif response.status_code == 200:
            return {
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'description': weather_data['weather'][0]['description']
            }
        else:
            return f"Error: {weather_data.get('message', 'Unknown error')} (Status code: {response.status_code})"
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

def calculate_macs_metrics(cuda_num=16384, freq=2700, MACs=1.52e9, 
                         batch_size=256, GPU_util=0.9, pic_num=60000):
    GPU_MACs_per_sec = cuda_num * freq * GPU_util * 10**6
    total_MACs = MACs * pic_num
    total_time = total_MACs / GPU_MACs_per_sec
    
    return GPU_MACs_per_sec, total_MACs, total_time

def main():
    # Calculate MACs metrics
    GPU_MACs_per_sec, total_MACs, total_time = calculate_macs_metrics()
    
    print("=== Performance Metrics ===")
    print(f"GPU_MACs_per_sec: {GPU_MACs_per_sec:,.2f}")
    print(f"total_MACs: {total_MACs:,.2f}")
    print(f"total_time: {total_time:.4f} seconds")
    
    # Get weather data
    api_key = '647626c3e259fcc8e1b5e8f16fa5dad4'  # Replace with your actual API key
    city = 'Turin'  # You can change the city
    
    print("\n=== Weather Information ===")
    weather = get_weather(api_key, city)
    if isinstance(weather, dict):
        print(f"City: {city}")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Description: {weather['description']}")
    else:
        print(f"Weather data error: {weather}")

if __name__ == "__main__":
    main()

