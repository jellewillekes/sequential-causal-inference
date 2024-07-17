import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import ssl
import certifi
import warnings


def get_city_coordinates(city_name, country_name='Germany'):
    geolocator = Nominatim(user_agent="UniqueAppNameOrPurpose",
                           ssl_context=ssl.create_default_context(cafile=certifi.where()))
    location = geolocator.geocode(f"{city_name}, {country_name}")

    if not location:
        parts = sorted(city_name.split(), key=len, reverse=True)
        for part in parts:
            location = geolocator.geocode(f"{part}, {country_name}")
            if location:
                break

    if location:
        return (location.latitude, location.longitude)
    else:
        warnings.warn(f"Coordinate for {city_name}, {country_name} not found!")
        return None


def calculate_distance(city1, city2, country_name='Germany'):
    coordinates1 = get_city_coordinates(city1, country_name)
    time.sleep(2)  # Pause for 2 seconds between geocoding requests
    coordinates2 = get_city_coordinates(city2, country_name)

    if coordinates1 and coordinates2:
        distance = geodesic(coordinates1, coordinates2).kilometers
        distance = round(distance)
        print(f"{city1} and {city2}:\t{distance} km")
        return distance
    else:
        if not coordinates1:
            warnings.warn(f"Coordinate for {city1}, {country_name} not found!")
        if not coordinates2:
            warnings.warn(f"Coordinate for {city2}, {country_name} not found!")
        return None



