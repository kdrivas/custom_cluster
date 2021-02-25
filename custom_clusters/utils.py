from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib
import matplotlib.pyplot as plt

import geopandas as gpd
    
def plot_points(lat, lng, centers, labels, n_clusters):
    fig, ax = plt.subplots(figsize=(8,8))

    types = []
    all_points = []
    for ix in range(n_clusters):
        temp = [Point(x, y) for lab, x, y in zip(labels, lat, lng) if lab == ix]
        all_points += temp
        types += [ix] * len(temp)

    temp = [Point(x, y) for lab, x, y in zip(labels, lat, lng) if lab == -1]
    all_points += temp
    types += [n_clusters] * len(temp)

    for ix in range(n_clusters):
        temp = [Point(x, y) for x, y in [centers[ix]]]
        all_points += temp
        types += [n_clusters + 1 + ix] * len(temp)

    results = pd.DataFrame()
    results['geometry'] = all_points
    results['types'] = types
    
    df_geo = gpd.GeoDataFrame(results, geometry=all_points)
    
    if -1 in list(set(types)):
        df_geo.loc[df_geo.types == n_clusters].geometry.plot(ax=ax, color="red", markersize=100)
    
    for ix in list(set(types)):
        if ix != n_clusters:
            df_geo.loc[df_geo.types == ix].geometry.plot(ax=ax, color="black", markersize=100)
            df_geo.loc[df_geo.types == n_clusters + 1 + ix].geometry.plot(ax=ax, color="black", markersize=100, marker='*')
    
    plt.plot()