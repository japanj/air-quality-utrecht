import json
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from osgeo import gdal
from osgeo import gdal_array as gdarr
from osgeo import ogr
import os
from matplotlib import colors as colors

def concat_month_file(path):
    """
    Create dataframe from csv files
    
    Args:
        path (string): folder path
    
    Returns:
        Dataframe that contains air particle measurement of year 2023
    """
    df = pd.DataFrame()
    count = 0
    for filename in os.listdir(path):
        temp_file = os.path.join(path, filename)
        temp_df = pd.read_csv(temp_file, delimiter=";", skiprows=9, encoding='unicode_escape')
        temp_df = temp_df.rename(columns={" Begindatumtijd": "Begindatumtijd"})
        if count == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])
        count += 1
    df = df.reset_index(drop=True)
    return df

def detect_outliers(data, col):
    """
    Detect outliers in each station/thing measurement data
    
    Args:
        data (dataframe): dataframe that contains air particle measurements
        col (string): Thing's name or Station code (column name)
    
    Returns:
        No value
    """
    q3 = np.nanquantile(data[col], 0.75)
    q1 = np.nanquantile(data[col], 0.25)
    IQR = q3 - q1
    lower_range = q1 - 1.5 * IQR
    upper_range = q3 + 1.5 * IQR
    outlier_list = [x for x in data[col] if ((x < lower_range) or (x > upper_range))]
    if len(outlier_list) != 0:
        print(col, "has outliers!")

def get_dup_stations(station_df, data_df):
    """
    Find duplicated stations and stations that need to be removed.
    
    Args:
        station_df (dataframe): dataframe contains station information
        data_df (dataframe): dataframe contains air particle measurements

    Returns:
        List of stations that need to be removed
    """
    dup_stations = station_df[station_df.duplicated(subset=['lat', 'lon'], keep=False)]
    # Store data points of each set of duplicated station in dictionary
    dup_count = {"name":[], "data points":[]}
    dup_list = list(dup_stations['name'])
    for col in data_df.columns:
        if col in dup_list:
            dup_count["name"].append(col)
            dup_count["data points"].append(data_df[col].count())
    dup_count_df = pd.DataFrame.from_dict(dup_count)
    # Keep the stations that has highest number of data points in each set of duplicated station
    dup_stations = pd.merge(dup_stations, dup_count_df, on="name", how="inner")
    keep_stations = dup_stations.sort_values(["data points"]).drop_duplicates(subset=['lat', 'lon'], keep="last")
    drop_stations = dup_stations[~dup_stations['name'].isin(keep_stations['name'])]
    drop_stations_list = list(drop_stations['name'])
    return drop_stations_list

def remove_column_outliers(datasource, df, drop_stations_list):
    """
    Remove the outliers (station) from dataframe.
    
    Args:
        datasource (string): Data source (sm for Samenmeten, lm for Luchtmeetnet)
        df (dataframe): Dataframe that contain air particle measurement
        drop_stations_list (list): List contains stations that need to be removed

    Returns:
        Two dataframes (one for annual average analysis and another one for peak and off-peak hour)
    """
    temp_df = df.copy()
    sensors_list = list(temp_df.columns)
    if datasource == "sm":
        sensors_list.remove('phenomenonTime')
        sensors_list.remove('phenomenonTime_datetime')
    threshold = round((5*len(temp_df))/100)
    drop_annual = []
    drop_month = []
    for col in sensors_list:
        # Drop the column that has value less than 5% of all data (annual average)
        # Drop the column that has value less than 100 data points (peak/off-peak hour average)
        if (temp_df[col].count() < threshold) or (col in drop_stations_list):
            drop_annual.append(col)
            drop_month.append(col)
        elif temp_df[col].count() < 100:
            drop_month.append(col)
    df_annual = temp_df.drop(columns=drop_annual)
    df_month = temp_df.drop(columns=drop_month)
    return df_annual, df_month

def get_air_particle_data(df, air_particle, start_date, end_date):
    """
    Get air particle data from Samenmeten.
    
    Args:
        df (dataframe): things information dataframe
        air_particle (string): air particle (e.g. no2, pm25)
        start_date (string): begin date in year-month-date format (e.g. 2023-01-01)
        end_date (string): end date in year-month-date format (e.g. 2023-12-31)
    
    Returns:
        List of json objects that contains air particle data of things
    """
    data = []
    column_name = "has_"+air_particle
    if air_particle == "pm25" or air_particle == "pm10":
        name_val = air_particle+"_kal"
    else:
        name_val = air_particle
    for index, row in df[df[column_name] == "yes"].iterrows():
        for item in row['Datastreams']:
            if item['ObservedProperty']['name'] == name_val:
                navigation_link = item['Observations@iot.navigationLink']
                print(row['name'], navigation_link)
                data_json = json.dumps(None)
                temp = json.dumps(None)
                while True:
                    try:
                        r = requests.get(navigation_link, params={
                            '$filter': "(date(phenomenonTime) ge date('"+start_date+"')) and (date(phenomenonTime) le date('"+end_date+"'))",
                            '$orderby': 'phenomenonTime'
                        })
                        data_json = json.loads(r.text)
                        if temp == "null":
                            if len(data_json['value']) != 0:
                                data_json['thingname'] = row['name']
                                data_json['lat'] = row['lat']
                                data_json['lon'] = row['lon']
                                temp = data_json
                        else:
                            for j in data_json['value']:
                                temp['value'].append(j)
                        if "@iot.nextLink" in data_json:
                            navigation_link = data_json['@iot.nextLink']
                        else:
                            break
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching data: {e}")
                        break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        break        
                if "@iot.nextLink" in temp:
                    del temp['@iot.nextLink']
                if temp != "null":
                    data.append(temp)
    return data

def create_air_particle_df(data):
    """
    Create dataframe that contains specific air particle data of each things
    
    Args:
        data (list): List of json objects that contains air particle data of things
    
    Returns:
        Dataframe that contains specific air particle data of each things
    """
    df = pd.json_normalize(data, record_path=['value'], meta=['thingname', 'lat', 'lon'])
    df = df[['thingname', 'lat', 'lon', 'phenomenonTime', 'result']]
    # Pivot the dataframe to be as same as the example and reset the index
    pivot_df = pd.pivot_table(df, values='result', index=['phenomenonTime'], columns=['thingname']).reset_index()
    # Rename the column axis (axis=1)
    # In this case, we set the column axis to be None
    pivot_df = pivot_df.rename_axis(None, axis=1)
    return pivot_df

def prepare_peak_offpeak_df(sm_month_df, lm_month_df, sm_station_df, lm_station_df):
    """
    Create geodataframe to be used in interpolation step.
    This function is specified for analyzing air particle measurement based on peak
    and off-peak hour.
    
    Args:
        sm_month_df (dataframe): dataframe that contains air particle measurement from Samenmeten
        lm_month_df (dataframe): dataframe that contains air particle measurement from Luchtmeetnet
        sm_station_df (dataframe): dataframe that contains things information from Samenmeten
        lm_station_df (dataframe): dataframe that contains things information from Luchtmeetnet
    
    Returns:
        Two geodataframes (one for peak hour and another for off-peak hour)
    """
    peak_hour_list = [6, 7, 8, 9, 16, 17, 18, 19]
    public_holiday_2023 = ["2023-01-01", "2023-04-07", "2023-04-09", 
                           "2023-04-10", "2023-04-27", "2023-05-05", 
                           "2023-05-18", "2023-05-28", "2023-05-29", 
                           "2023-12-25", "2023-12-26"]
    new_sm_month_df = sm_month_df.rename(columns={'phenomenonTime_datetime': 'datetime_col'})
    new_lm_df = lm_month_df.rename(columns={'begin_datetime': 'datetime_col'})
    month_df = pd.merge(new_sm_month_df, new_lm_df, on="datetime_col", how="inner")
    month_df['weekday'] = month_df['datetime_col'].dt.day_name()
    month_df['num_weekday'] = month_df['datetime_col'].dt.day_of_week
    month_df['hour'] = month_df['datetime_col'].dt.hour
    month_df['date'] = month_df['datetime_col'].dt.strftime('%Y-%m-%d')
    # peak df
    mean_peak_hour_df = month_df[(month_df['num_weekday'] < 5) & (month_df['hour'].isin(peak_hour_list)) & (~month_df['date'].isin(public_holiday_2023))].mean(numeric_only=True).to_frame().reset_index()
    mean_peak_hour_df = mean_peak_hour_df.rename(columns={'index':'name', 0:'value'})
    mean_peak_hour_df = mean_peak_hour_df[(mean_peak_hour_df['name'] != "num_weekday") & (mean_peak_hour_df['name'] != "hour")]
    # off-peak df
    mean_offpeak_hour_df = month_df[~((month_df['num_weekday'] < 5) & (month_df['hour'].isin(peak_hour_list)) & (~month_df['date'].isin(public_holiday_2023)))].mean(numeric_only=True).to_frame().reset_index()
    mean_offpeak_hour_df = mean_offpeak_hour_df.rename(columns={'index':'name', 0:'value'})
    mean_offpeak_hour_df = mean_offpeak_hour_df[(mean_offpeak_hour_df['name'] != "num_weekday") & (mean_offpeak_hour_df['name'] != "hour")]
    lm_station_df_temp = lm_station_df.rename(columns={'StationsCode':'name'})
    temp_df_list = []
    for i in range(2):
        if i == 0:
            all_mean_df = mean_peak_hour_df
        else:
            all_mean_df = mean_offpeak_hour_df
        all_mean_df = pd.merge(all_mean_df, sm_station_df, on="name", how="left", suffixes=("_ori", "_sm"))
        all_mean_df = pd.merge(all_mean_df, lm_station_df_temp, on="name", how="left", suffixes=("_ori", "_lm"))
        for index, row in all_mean_df.iterrows():
            if "NL" in row['name']:
                all_mean_df.loc[index, 'lat_ori'] = row['lat_lm']
                all_mean_df.loc[index, 'lon_ori'] = row['lon_lm']
        all_mean_gdf = (gpd.GeoDataFrame(all_mean_df, crs="EPSG:4326", geometry=gpd.points_from_xy(all_mean_df['lon_ori'], all_mean_df['lat_ori'])).to_crs("EPSG:28992"))
        all_mean_gdf['Easting'] = all_mean_gdf.geometry.x
        all_mean_gdf['Northing'] = all_mean_gdf.geometry.y
        temp_df_list.append(all_mean_gdf)
    mean_peak_hour_gdf = temp_df_list[0].copy()
    mean_offpeak_hour_gdf = temp_df_list[1].copy()
    return mean_peak_hour_gdf, mean_offpeak_hour_gdf

def interpolate_value(gdf, output_path):
    """
    Interpolate air particle measurement values and 
    save the interpolation result as a raster.
    
    Args:
        gdf (geodataframe): geodataframe that is prepared for interpolation step
        output_path (string): raster file name  
    
    Returns:
        No value
    """
    resolution = 30  # Cell size in meters
    grid_x = np.arange(gdf.bounds.minx.min(), gdf.bounds.maxx.max(), resolution)
    grid_y = np.arange(gdf.bounds.miny.min(), gdf.bounds.maxy.max(), resolution)
    # model = NearestNDInterpolator(list(zip(gdf['Easting'], gdf['Northing'])), gdf['value'])
    model = LinearNDInterpolator(list(zip(gdf['Easting'], gdf['Northing'])), gdf['value'])
    interpolate_val = model(*np.meshgrid(grid_x, grid_y))
    # Inverse row order (if not revert it, result gonna be wrong when display with the municipality boundary)
    new_interpolate_val = interpolate_val[::-1]
    # Visualization
    plt.pcolormesh(grid_x, grid_y, interpolate_val, shading='auto')
    plt.plot(gdf['Easting'], gdf['Northing'], "ok", label="input point")
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.title("Interpolation result")
    plt.show()
    # Define GeoTIFF metadata
    width = new_interpolate_val.shape[1]
    height = new_interpolate_val.shape[0]
    # Calculate the pixel size in both x and y directions
    pixel_size_x = (grid_x.max() - grid_x.min()) / width
    pixel_size_y = (grid_y.min() - grid_y.max()) / height
    # Create the transformation
    transform = Affine(pixel_size_x, 0, grid_x.min(), 0, pixel_size_y, grid_y.max())
    # Save the raster as a GeoTIFF file
    with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs=gdf.crs, transform=transform) as dst:
        dst.write(new_interpolate_val, 1)
        dst.close()

def prepare_bikepath_raster_data(input_path, output_path):
    """
    Return air particle measurements along bike path.
    
    Args:
        input_path (string): input file name
        output_path (string): output file name
    
    Returns:
        Air particle measurements along bike path raster in form of numpy array
    """
    raster_ds = gdal.Warp(output_path,
                          input_path,
                          format="GTiff",
                          dstSRS="EPSG:4326",
                          cutlineDSName="data/utrecht_boundary.geojson",
                          dstNodata=-9999,
                          cropToCutline=True,
                          outputType=gdal.GDT_Float32)
    raster_ds = None
    bikepath_vector_ds = ogr.Open("data/utrecht_bikepath.geojson")
    bikepath_layer = bikepath_vector_ds.GetLayer()
    # create empty raster
    driver = gdal.GetDriverByName('Mem')
    raster = gdal.Open(output_path)
    bikepath_raster_ds = driver.Create('', raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Float32)
    bikepath_raster_ds.SetProjection(raster.GetProjection())
    bikepath_raster_ds.SetGeoTransform(raster.GetGeoTransform())
    band = bikepath_raster_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    # Rasterize bike path vector
    gdal.RasterizeLayer(bikepath_raster_ds,
                        [1],
                        bikepath_layer,
                        burn_values=[1], 
                        options=['ALL_TOUCHED=FALSE'])

    bikepath_arr = gdarr.DatasetReadAsArray(bikepath_raster_ds, 0, 0, bikepath_raster_ds.RasterXSize, bikepath_raster_ds.RasterYSize)
    raster_arr = gdarr.DatasetReadAsArray(raster, 0, 0, raster.RasterXSize, raster.RasterYSize)
    bikepath_arr[bikepath_arr == 0] = None
    bikepath_data = bikepath_arr * raster_arr
    return bikepath_data

def prepare_refbound_raster(input_path, output_path):
    """
    Create one raster as a boundary reference for visualization 
    
    Args:
        input_path (string): input file name
        output_path (string): output file name
    
    Returns:
        No value
    """
    refbound_raster_ds = gdal.Warp(output_path,
                                   input_path,
                                   format="GTiff",
                                   dstSRS="EPSG:4326",
                                   cutlineDSName="data/utrecht_boundary.geojson",
                                   dstNodata=-9999,
                                   cropToCutline=True,
                                   outputType=gdal.GDT_Float32)
    refbound_raster_ds = None

def prepare_avg_annual_df(sm_annual_df, lm_annual_df, sm_station_df, lm_station_df):
    """
    Create geodataframe to be used in interpolation step.
    This function is specified for analyzing annual average air particle measurement.
    
    Args:
        sm_annual_df (dataframe): dataframe that contains air particle measurement from Samenmeten
        lm_annual_df (dataframe): dataframe that contains air particle measurement from Luchtmeetnet
        sm_station_df (dataframe): dataframe that contains things information from Samenmeten
        lm_station_df (dataframe): dataframe that contains things information from Luchtmeetnet
    
    Returns:
        Geodataframe used for interpolation step
    """
    sm_mean = sm_annual_df.mean(numeric_only=True)
    lm_mean = lm_annual_df.mean(numeric_only=True)
    all_mean_df = pd.concat([sm_mean, lm_mean]).to_frame().reset_index()
    all_mean_df = all_mean_df.rename(columns={'index':'name', 0:'value'})
    lm_station_df_temp = lm_station_df.rename(columns={'StationsCode':'name'})
    all_mean_df = pd.merge(all_mean_df, sm_station_df, on="name", how="left", suffixes=("_ori", "_sm"))
    all_mean_df = pd.merge(all_mean_df, lm_station_df_temp, on="name", how="left", suffixes=("_ori", "_lm"))
    for index, row in all_mean_df.iterrows():
        if "NL" in row['name']:
            all_mean_df.loc[index, 'lat_ori'] = row['lat_lm']
            all_mean_df.loc[index, 'lon_ori'] = row['lon_lm']
    all_mean_gdf = (gpd.GeoDataFrame(all_mean_df, crs="EPSG:4326", geometry=gpd.points_from_xy(all_mean_df['lon_ori'], all_mean_df['lat_ori'])).to_crs("EPSG:28992"))
    all_mean_gdf['Easting'] = all_mean_gdf.geometry.x
    all_mean_gdf['Northing'] = all_mean_gdf.geometry.y
    return all_mean_gdf

def map_value_with_color(value, colormap):
    """
    Map pixel value to color in colormap
    
    Args:
        value (float): pixel value 
        colormap (colormap): linear colormap

    Returns:
        Color value in rgba format
    """
    if np.isnan(value):
        return (1, 0, 0, 0)
    else:
        return colors.to_rgba(colormap(value), 1)  