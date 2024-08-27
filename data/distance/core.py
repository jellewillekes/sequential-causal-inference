import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import ssl
import certifi
import warnings


def get_city_coordinates(team_name, city_name, country_name='Germany'):
    geolocator = Nominatim(user_agent="UniqueAppNameOrPurpose",
                           ssl_context=ssl.create_default_context(cafile=certifi.where()))

    # Try to geocode using the full team name and city name
    location = geolocator.geocode(f"{team_name} {city_name} {country_name}")

    # If no location is found, try using only the city name and country name
    if not location:
        location = geolocator.geocode(f"{city_name} {country_name}")

    # If a location is found, return the coordinates
    if location:
        return (location.latitude, location.longitude)
    else:
        warnings.warn(f"Coordinate for {city_name}, {country_name} not found!")
        return None


def calculate_distance(team1, team2, city1, city2, country_name='Germany'):
    coordinates1 = get_city_coordinates(team1, city1, country_name)
    time.sleep(2)  # Pause for 2 seconds between geocoding requests
    coordinates2 = get_city_coordinates(team2, city2, country_name)

    if coordinates1 and coordinates2:
        distance = geodesic(coordinates1, coordinates2).kilometers
        distance = round(distance)
        print(f"{team1} {city1} and {team2} {city2}:\t{distance} km")
        return distance
    else:
        if not coordinates1:
            warnings.warn(f"Coordinate for {city1}, {country_name} not found!")
        if not coordinates2:
            warnings.warn(f"Coordinate for {city2}, {country_name} not found!")
        return None



