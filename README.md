# Air quality analysis in Utrecht municipality
Analyze the air quality in Utrecht by using data from Luchtmeetnet and Samenmeten.

This project is part of Acquisition and Exploration of Geo-Data course.

## Folder Structure
* chart (contains all the charts)
* data (contains data used for analysis)
    * luchtmeetnet (data from luchtmeetnet)
    * utrecht-latest (road networks shape file)
* interpolation (contains interpolation result)
* map (contains interactive map visualization of air pollution data)
* utils (contains custom functions)

main file is *aq_utrecht_project.ipynb*

## Dataset
This analysis contained the dataset as desribed in the below table.
| Data | Description | Owner |
|------|-------------|-------|
|Utrecht boundary|Utrecht boundary in JSON format|PDOK|
|Utrecht subarea boundary|Utrecht subarea boundary in JSON format|PDOK|
|Bike path in Utrecht|Bike path in Utrecht province in shapefile format (data from OpenStreetMap)|Geofabrik|
|Land use in Utrecht|Existing land use in Utrecht in JSON format|PDOK|
|Things (stations) metadata in Utrecht|Things metadata in form of JSON format|Samenmeten|
|Official stations metadata in Utrecht|Official stations metadata from CSV file|Luchtmeetnet|
|PM 2.5 and NO2 measurements|PM 2.5 and NO2 measurements data in JSON format|Samenmeten|
|PM 2.5 and NO2 measurements|PM 2.5 and NO2 measurements data in CSV file (each CSV file contains data for each month)|Luchtmeetnet|

## Required Libraries
``` python
import geopandas as gpd
import json 
import requests
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
import branca.colormap as cm
from matplotlib import colors as colors
from utils.utils import *
import scipy
import seaborn as sns
```