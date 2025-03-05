import openmeteo_requests

import requests
import requests_cache
import pandas as pd
from retry_requests import retry

def get_weather_forecast_data(latitude,longitude,start_date,end_date):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"start_date": start_date,
		"end_date": end_date,
		"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"],
		"wind_speed_unit": "ms",
		"timezone": "Europe/London"
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["precipitation"] = hourly_precipitation
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	return hourly_dataframe

def get_weather_data(latitude,longitude,start_date,end_date):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"start_date": start_date,
		"end_date": end_date,
		"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"],
		"wind_speed_unit": "ms",
		"timezone": "Europe/London"
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["precipitation"] = hourly_precipitation
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	return hourly_dataframe

def get_coords(city_name):
	url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json"
	response = requests.get(url)
	R={}
	R['latitude']=response.json()['results'][0]['latitude']
	R['longitude']=response.json()['results'][0]['longitude']
	return R

def create_df(city_name,start_date,end_date):
	coords=get_coords(city_name)
	df=get_weather_data(coords['latitude'],coords['longitude'],start_date,end_date)
	df_city=pd.DataFrame({'city':[city_name]*df.shape[0]})
	df_concat = pd.concat([df_city, df], axis=1)
	df_concat.set_index(['city', 'date'], inplace=True)
	return df_concat

if __name__=='__main__':
	print(create_df('Tours',"2024-01-01","2024-12-31"))
#print(get_coord('Tours'))
#"2024-01-01"
#"2024-12-31"
