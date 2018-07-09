

```python
#! pip install citipy
#! pip install unidecode
```


```python
# Dependencies and Setup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests as req
import time
import csv
import random
import requests
import api_keys
import math
import os


# Incorporated citipy to determine city based on latitude and longitude
from citipy import citipy

# Output File (CSV)
output_data_file = "output_data/cities.csv"

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)
```


```python
lat_lngs = []
cities = []
Country = []
 
lats = np.random.uniform(low=-90.000, high=90.000, size=1500)
lngs = np.random.uniform(low=-180.000, high=180.000, size=1500)
lat_lngs = zip(lats, lngs)

for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    country = citipy.nearest_city(lat_lng[0], lat_lng[1]).country_code
  
    if city not in cities:
        cities.append(city)
        Country.append(country)

len(cities)
```




    578




```python
cities = pd.DataFrame()
cities['rand_lat'] = lats
cities['rand_lng'] = lngs

for index, row in cities.iterrows():
    lat = row['rand_lat']
    lng = row['rand_lng']
    cities['closest_city'] = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    cities['country'] = citipy.nearest_city(lat_lng[0], lat_lng[1]).country_code
#location = location.drop_duplicates(['closest_city', 'country'])
#location = location.dropna()
#len(location['closest_city'].value_counts())

cities.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rand_lat</th>
      <th>rand_lng</th>
      <th>closest_city</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-46.918913</td>
      <td>174.473402</td>
      <td>albany</td>
      <td>au</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.056764</td>
      <td>121.320803</td>
      <td>albany</td>
      <td>au</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27.565546</td>
      <td>4.733968</td>
      <td>albany</td>
      <td>au</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-50.608187</td>
      <td>111.951000</td>
      <td>albany</td>
      <td>au</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.008478</td>
      <td>149.822527</td>
      <td>albany</td>
      <td>au</td>
    </tr>
  </tbody>
</table>
</div>




```python
citypy = os.path.join("worldcities.csv")
citipy_CSV = pd.read_csv(citypy)
citypy_CSV = citipy_CSV.sample(n=505)
citypy_CSV.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>City</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45480</th>
      <td>us</td>
      <td>mauldin</td>
      <td>34.778611</td>
      <td>-82.310278</td>
    </tr>
    <tr>
      <th>22709</th>
      <td>ly</td>
      <td>zlitan</td>
      <td>32.466667</td>
      <td>14.566667</td>
    </tr>
    <tr>
      <th>8874</th>
      <td>de</td>
      <td>gartringen</td>
      <td>48.650000</td>
      <td>8.900000</td>
    </tr>
    <tr>
      <th>34550</th>
      <td>ro</td>
      <td>turburea</td>
      <td>44.716667</td>
      <td>23.516667</td>
    </tr>
    <tr>
      <th>44025</th>
      <td>us</td>
      <td>bay city</td>
      <td>43.594444</td>
      <td>-83.888889</td>
    </tr>
  </tbody>
</table>
</div>




```python
count = 0
api_key = api_keys.api_key
url = "http://api.openweathermap.org/data/2.5/weather?units=Imperial&APPID=" + api_key 
units = "imperial" 
query_url = url + "lat=" + str(row["Latitude"]) + "&lon=" + str(row["Longitude"]) +"&units=" + units

citypy_CSV["Temperature"] = ""
citypy_CSV["Humidity"] = ""
citypy_CSV["Cloudiness"] = ""
citypy_CSV["Wind Speed"] = ""

for index, row in citypy_CSV.iterrows():
    time.sleep(5)
    count += 1
    weather = requests.get(query_url).json()
    temperature = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]
    cloudiness = weather["clouds"]["all"]
    windSpeed = weather["wind"]["speed"]
    
    
    citypy_CSV.set_value(index, "Temperature", temperature)
    citypy_CSV.set_value(index, "Humidity", humidity)
    citypy_CSV.set_value(index, "Cloudiness", cloudiness)
    citypy_CSV.set_value(index, "Wind Speed", windSpeed)
citypy_CSV
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-8-d641f3bcf2e2> in <module>()
         14     count += 1
         15     weather = requests.get(query_url).json()
    ---> 16     temperature = weather["main"]["temp"]
         17     citypy_CSV.set_value(index, "Temperature", temperature)
         18 
    

    KeyError: 'main'

