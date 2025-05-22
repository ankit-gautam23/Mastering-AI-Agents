import pytest
from src.weather_assistant import WeatherAssistant

@pytest.fixture
def weather_assistant():
    return WeatherAssistant()

def test_get_current_weather(weather_assistant):
    # TODO: Implement test for get_current_weather
    # 1. Test with valid location
    # 2. Test with invalid location
    # 3. Test error handling
    pass

def test_get_forecast(weather_assistant):
    # TODO: Implement test for get_forecast
    # 1. Test with valid location and days
    # 2. Test with invalid location
    # 3. Test with invalid days
    pass

def test_convert_units(weather_assistant):
    # TODO: Implement test for convert_units
    # 1. Test temperature conversion
    # 2. Test distance conversion
    # 3. Test speed conversion
    # 4. Test invalid units
    pass

def test_handle_query(weather_assistant):
    # TODO: Implement test for handle_query
    # 1. Test weather query
    # 2. Test forecast query
    # 3. Test unit conversion query
    # 4. Test invalid query
    pass 