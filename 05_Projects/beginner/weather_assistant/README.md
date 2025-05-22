# Weather Assistant

A simple weather assistant agent that can provide weather information and forecasts based on user queries.

## Project Overview

This project implements a weather assistant that can:
- Provide current weather conditions
- Give weather forecasts
- Handle location-based queries
- Convert between temperature units
- Provide weather alerts

## Requirements

### Functional Requirements
1. Weather Information
   - Current temperature
   - Weather conditions
   - Humidity and wind
   - UV index
   - Precipitation chance

2. Location Handling
   - City name recognition
   - Coordinates lookup
   - Time zone handling
   - Location validation

3. Unit Conversion
   - Celsius to Fahrenheit
   - Kilometers to Miles
   - Metric to Imperial

4. Error Handling
   - Invalid locations
   - API failures
   - Network issues
   - Invalid inputs

### Technical Requirements
1. Implement the following classes:
   - WeatherAssistant
   - LocationHandler
   - UnitConverter
   - WeatherAPI

2. Write unit tests
3. Implement error handling
4. Add logging
5. Create documentation

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Complete the TODO items in the code
4. Run tests:
   ```bash
   pytest tests/
   ```

## Code Structure

```
weather_assistant/
├── src/
│   ├── __init__.py
│   ├── weather_assistant.py
│   ├── location_handler.py
│   ├── unit_converter.py
│   └── weather_api.py
├── tests/
│   ├── __init__.py
│   ├── test_weather_assistant.py
│   ├── test_location_handler.py
│   ├── test_unit_converter.py
│   └── test_weather_api.py
├── requirements.txt
└── README.md
```

## Implementation Tasks

### 1. Weather Assistant
```python
class WeatherAssistant:
    def __init__(self):
        self.location_handler = LocationHandler()
        self.unit_converter = UnitConverter()
        self.weather_api = WeatherAPI()

    def get_current_weather(self, location):
        # TODO: Implement current weather retrieval
        pass

    def get_forecast(self, location, days):
        # TODO: Implement weather forecast
        pass

    def convert_units(self, value, from_unit, to_unit):
        # TODO: Implement unit conversion
        pass

    def handle_query(self, query):
        # TODO: Implement query handling
        pass
```

### 2. Location Handler
```python
class LocationHandler:
    def __init__(self):
        self.locations = {}
        self.coordinates = {}

    def validate_location(self, location):
        # TODO: Implement location validation
        pass

    def get_coordinates(self, location):
        # TODO: Implement coordinate lookup
        pass

    def get_timezone(self, location):
        # TODO: Implement timezone lookup
        pass
```

### 3. Unit Converter
```python
class UnitConverter:
    def __init__(self):
        self.conversion_rates = {}

    def convert_temperature(self, value, from_unit, to_unit):
        # TODO: Implement temperature conversion
        pass

    def convert_distance(self, value, from_unit, to_unit):
        # TODO: Implement distance conversion
        pass

    def convert_speed(self, value, from_unit, to_unit):
        # TODO: Implement speed conversion
        pass
```

### 4. Weather API
```python
class WeatherAPI:
    def __init__(self):
        self.api_key = None
        self.base_url = None

    def get_current_weather(self, location):
        # TODO: Implement API call for current weather
        pass

    def get_forecast(self, location, days):
        # TODO: Implement API call for forecast
        pass

    def get_alerts(self, location):
        # TODO: Implement API call for alerts
        pass
```

## Expected Output

```
User: What's the weather in New York?
Assistant: Current weather in New York:
- Temperature: 72°F (22°C)
- Conditions: Partly Cloudy
- Humidity: 65%
- Wind: 8 mph (13 km/h)
- UV Index: Moderate

User: Will it rain tomorrow?
Assistant: Forecast for New York tomorrow:
- High: 75°F (24°C)
- Low: 65°F (18°C)
- Precipitation: 30% chance
- Conditions: Cloudy with light rain expected

User: Convert 32°C to Fahrenheit
Assistant: 32°C is equal to 89.6°F
```

## Learning Objectives

By completing this project, you will learn:
1. API integration
2. Data parsing and formatting
3. Unit conversion
4. Error handling
5. Location services
6. Testing and documentation

## Resources

### Documentation
- [Python Documentation](https://docs.python.org/3/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Geocoding API](https://developers.google.com/maps/documentation/geocoding)

### Tools
- [Python](https://www.python.org/)
- [Pytest](https://docs.pytest.org/)
- [VS Code](https://code.visualstudio.com/)

### Learning Materials
- [API Integration](https://realpython.com/api-integration-in-python/)
- [Python Testing](https://realpython.com/python-testing/)
- [Error Handling](https://realpython.com/python-exceptions/)

## Evaluation Criteria

Your implementation will be evaluated based on:
1. Code Quality
   - Clean and well-documented code
   - Proper error handling
   - Efficient algorithms
   - Good test coverage

2. Functionality
   - Accurate weather information
   - Proper unit conversion
   - Location handling
   - Error handling

3. Documentation
   - Clear README
   - Code comments
   - API documentation
   - Test documentation

## Submission

1. Complete the implementation
2. Write tests for all components
3. Document your code
4. Create a pull request

## Next Steps

After completing this project, you can:
1. Add more weather data sources
2. Implement weather alerts
3. Add historical weather data
4. Create a web interface
5. Add weather visualization
6. Implement caching 