import pytest
from src.unit_converter import UnitConverter

@pytest.fixture
def unit_converter():
    return UnitConverter()

def test_convert_temperature(unit_converter):
    # TODO: Implement test for convert_temperature
    # 1. Test celsius to fahrenheit
    # 2. Test fahrenheit to celsius
    # 3. Test invalid units
    pass

def test_convert_distance(unit_converter):
    # TODO: Implement test for convert_distance
    # 1. Test km to miles
    # 2. Test miles to km
    # 3. Test invalid units
    pass

def test_convert_speed(unit_converter):
    # TODO: Implement test for convert_speed
    # 1. Test kmh to mph
    # 2. Test mph to kmh
    # 3. Test invalid units
    pass

def test_validate_units(unit_converter):
    # TODO: Implement test for validate_units
    # 1. Test valid temperature units
    # 2. Test valid distance units
    # 3. Test valid speed units
    # 4. Test invalid units
    pass 