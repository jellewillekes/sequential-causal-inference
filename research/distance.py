from geopy.geocoders import Nominatim
from geopy.distance import geodesic


def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="UniqueAppNameOrPurpose")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None


def calculate_distance(city1, city2):
    coordinates1 = get_city_coordinates(city1)
    coordinates2 = get_city_coordinates(city2)

    if coordinates1 and coordinates2:
        distance = geodesic(coordinates1, coordinates2).kilometers
        return distance
    else:
        return None


# Example usage
city1 = "‘s-Gravenhage"
city2 = "De Koog"
distance = calculate_distance(city1, city2)
print(f"The distance between {city1} and {city2} is {distance} kilometers.")
