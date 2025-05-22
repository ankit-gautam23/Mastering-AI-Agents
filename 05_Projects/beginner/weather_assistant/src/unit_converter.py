from typing import Dict, Optional

class UnitConverter:
    def __init__(self):
        self.conversion_rates = {
            'temperature': {
                'celsius_to_fahrenheit': lambda x: (x * 9/5) + 32,
                'fahrenheit_to_celsius': lambda x: (x - 32) * 5/9
            },
            'distance': {
                'km_to_miles': lambda x: x * 0.621371,
                'miles_to_km': lambda x: x / 0.621371
            },
            'speed': {
                'kmh_to_mph': lambda x: x * 0.621371,
                'mph_to_kmh': lambda x: x / 0.621371
            }
        }

    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between units.
        
        Args:
            value: Temperature value
            from_unit: Source unit ('celsius' or 'fahrenheit')
            to_unit: Target unit ('celsius' or 'fahrenheit')
            
        Returns:
            Converted temperature
        """
        # TODO: Implement temperature conversion
        # 1. Validate units
        # 2. Get conversion function
        # 3. Apply conversion
        pass

    def convert_distance(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert distance between units.
        
        Args:
            value: Distance value
            from_unit: Source unit ('km' or 'miles')
            to_unit: Target unit ('km' or 'miles')
            
        Returns:
            Converted distance
        """
        # TODO: Implement distance conversion
        # 1. Validate units
        # 2. Get conversion function
        # 3. Apply conversion
        pass

    def convert_speed(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert speed between units.
        
        Args:
            value: Speed value
            from_unit: Source unit ('kmh' or 'mph')
            to_unit: Target unit ('kmh' or 'mph')
            
        Returns:
            Converted speed
        """
        # TODO: Implement speed conversion
        # 1. Validate units
        # 2. Get conversion function
        # 3. Apply conversion
        pass

    def validate_units(self, unit_type: str, from_unit: str, to_unit: str) -> bool:
        """
        Validate if units are supported.
        
        Args:
            unit_type: Type of unit ('temperature', 'distance', 'speed')
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            True if units are valid, False otherwise
        """
        # TODO: Implement unit validation
        # 1. Check if unit type exists
        # 2. Check if units are supported
        # 3. Return validation result
        pass 