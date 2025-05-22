import pytest
from src.weather_api import WeatherAPI

@pytest.fixture
def weather_api():
    return WeatherAPI()

def test_get_current_weather(weather_api):
    # TODO: Implement test for get_current_weather
    # 1. Test with valid location
    # 2. Test with invalid location
    # 3. Test API error handling
    pass

def test_get_forecast(weather_api):
    # TODO: Implement test for get_forecast
    # 1. Test with valid location and days
    # 2. Test with invalid location
    # 3. Test with invalid days
    # 4. Test API error handling
    pass

def test_get_alerts(weather_api):
    # TODO: Implement test for get_alerts
    # 1. Test with valid location
    # 2. Test with invalid location
    # 3. Test API error handling
    pass

def test_make_request(weather_api):
    # TODO: Implement test for _make_request
    # 1. Test successful request
    # 2. Test invalid endpoint
    # 3. Test invalid parameters
    # 4. Test API error handling
    pass 