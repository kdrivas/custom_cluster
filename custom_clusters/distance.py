import arcgis
import pandas as pd
from arcgis.gis import GIS

def get_time_function(route):
    def get_time_distance(point:list, centers:list):
        distances = []
        for center in centers:
            stops = f"{point[1]},{point[0]};{center[1]},{center[0]}"

            result = route.solve(stops=stops,
                                    start_time="now",
                                    return_directions=False,
                                    directions_language="es",)

            time = result['routes']['features'][0]['attributes']['Total_TravelTime']
            distances.append(time)
            
        return distances
    
    return get_time_distance