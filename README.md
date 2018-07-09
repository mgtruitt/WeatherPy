

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




    613




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
      <td>45.183919</td>
      <td>24.777809</td>
      <td>saint-francois</td>
      <td>gp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.430072</td>
      <td>147.551405</td>
      <td>saint-francois</td>
      <td>gp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.666475</td>
      <td>-143.628799</td>
      <td>saint-francois</td>
      <td>gp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57.913970</td>
      <td>107.628348</td>
      <td>saint-francois</td>
      <td>gp</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.195941</td>
      <td>-76.650104</td>
      <td>saint-francois</td>
      <td>gp</td>
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
      <th>3614</th>
      <td>br</td>
      <td>pedra</td>
      <td>-8.500000</td>
      <td>-36.950000</td>
    </tr>
    <tr>
      <th>13341</th>
      <td>gr</td>
      <td>aitolikon</td>
      <td>38.433333</td>
      <td>21.350000</td>
    </tr>
    <tr>
      <th>38501</th>
      <td>ru</td>
      <td>tatarka</td>
      <td>44.958895</td>
      <td>41.951712</td>
    </tr>
    <tr>
      <th>41524</th>
      <td>ua</td>
      <td>neresnytsya</td>
      <td>48.118145</td>
      <td>23.765878</td>
    </tr>
    <tr>
      <th>10888</th>
      <td>es</td>
      <td>los barrios</td>
      <td>36.184704</td>
      <td>-5.489539</td>
    </tr>
  </tbody>
</table>
</div>




```python
count = 0
api_key = api_keys.api_key
url = "http://api.openweathermap.org/data/2.5/weather?APPID=" + api_key 
units = "imperial" 

citypy_CSV["Temperature"] = ""
citypy_CSV["Humidity"] = ""
citypy_CSV["Cloudiness"] = ""
citypy_CSV["Wind Speed"] = ""

for index, row in citypy_CSV.iterrows():
    time.sleep(5)
    query_url = url + units + "lat=" + str(row["Latitude"]) + "&lon=" + str(row["Longitude"]) +"&units="
    count += 1
    weather = requests.get(query_url + citypy_CSV).json()
    temperature = weather["list"]["main"]["temp"]
    humidity = weather[list]["main"]["humidity"]
    cloudiness = weather["clouds"]["all"]
    windSpeed = weather["wind"]["speed"]
    
    
    citypy_CSV.set_value(index, "Temperature", temperature)
    citypy_CSV.set_value(index, "Humidity", humidity)
    citypy_CSV.set_value(index, "Cloudiness", cloudiness)
    citypy_CSV.set_value(index, "Wind Speed", windSpeed)
citypy_CSV
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in eval(self, func, other, errors, try_cast, mgr)
       1317             values, values_mask, other, other_mask = self._try_coerce_args(
    -> 1318                 transf(values), other)
       1319         except TypeError:
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in _try_coerce_args(self, values, other)
        700                 type(other).__name__,
    --> 701                 type(self).__name__.lower().replace('Block', '')))
        702 
    

    TypeError: cannot convert str to an floatblock

    
    During handling of the above exception, another exception occurred:
    

    TypeError                                 Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in na_op(x, y)
       1201         try:
    -> 1202             result = expressions.evaluate(op, str_rep, x, y, **eval_kwargs)
       1203         except TypeError:
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in evaluate(op, op_str, a, b, use_numexpr, **eval_kwargs)
        203     if use_numexpr:
    --> 204         return _evaluate(op, op_str, a, b, **eval_kwargs)
        205     return _evaluate_standard(op, op_str, a, b)
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_numexpr(op, op_str, a, b, truediv, reversed, **eval_kwargs)
        118     if result is None:
    --> 119         result = _evaluate_standard(op, op_str, a, b)
        120 
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_standard(op, op_str, a, b, **eval_kwargs)
         63     with np.errstate(all='ignore'):
    ---> 64         return op(a, b)
         65 
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in <lambda>(x, y)
         77                          default_axis=default_axis),
    ---> 78         radd=arith_method(lambda x, y: y + x, names('radd'), op('+'),
         79                           default_axis=default_axis),
    

    TypeError: must be str, not float

    
    During handling of the above exception, another exception occurred:
    

    TypeError                                 Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in eval(self, func, other, errors, try_cast, mgr)
       1376             with np.errstate(all='ignore'):
    -> 1377                 result = get_result(other)
       1378 
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in get_result(other)
       1345             else:
    -> 1346                 result = func(values, other)
       1347 
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in na_op(x, y)
       1227                     with np.errstate(all='ignore'):
    -> 1228                         result[mask] = op(xrav, y)
       1229             else:
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in <lambda>(x, y)
         77                          default_axis=default_axis),
    ---> 78         radd=arith_method(lambda x, y: y + x, names('radd'), op('+'),
         79                           default_axis=default_axis),
    

    TypeError: must be str, not float

    
    During handling of the above exception, another exception occurred:
    

    TypeError                                 Traceback (most recent call last)

    <ipython-input-17-6f315c9093d6> in <module>()
         13     query_url = url + units + "lat=" + str(row["Latitude"]) + "&lon=" + str(row["Longitude"]) +"&units="
         14     count += 1
    ---> 15     weather = requests.get(query_url + citypy_CSV).json()
         16     temperature = weather["list"]["main"]["temp"]
         17     humidity = weather[list]["main"]["humidity"]
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in f(self, other, axis, level, fill_value)
       1265                 self = self.fillna(fill_value)
       1266 
    -> 1267             return self._combine_const(other, na_op)
       1268 
       1269     f.__name__ = name
    

    ~\Anaconda3\lib\site-packages\pandas\core\frame.py in _combine_const(self, other, func, errors, try_cast)
       3985         new_data = self._data.eval(func=func, other=other,
       3986                                    errors=errors,
    -> 3987                                    try_cast=try_cast)
       3988         return self._constructor(new_data)
       3989 
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in eval(self, **kwargs)
       3433 
       3434     def eval(self, **kwargs):
    -> 3435         return self.apply('eval', **kwargs)
       3436 
       3437     def quantile(self, **kwargs):
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in apply(self, f, axes, filter, do_integrity_check, consolidate, **kwargs)
       3327 
       3328             kwargs['mgr'] = self
    -> 3329             applied = getattr(b, f)(**kwargs)
       3330             result_blocks = _extend_blocks(applied, result_blocks)
       3331 
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in eval(self, func, other, errors, try_cast, mgr)
       1321             return block.eval(func, orig_other,
       1322                               errors=errors,
    -> 1323                               try_cast=try_cast, mgr=mgr)
       1324 
       1325         # get the result, may need to transpose the other
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in eval(self, func, other, errors, try_cast, mgr)
       1382             raise
       1383         except Exception as detail:
    -> 1384             result = handle_error()
       1385 
       1386         # technically a broadcast error in numpy can 'work' by returning a
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals.py in handle_error()
       1365                 # The 'detail' variable is defined in outer scope.
       1366                 raise TypeError('Could not operate %s with block values %s' %
    -> 1367                                 (repr(other), str(detail)))  # noqa
       1368             else:
       1369                 # return the values
    

    TypeError: Could not operate 'http://api.openweathermap.org/data/2.5/weather?APPID=922e60cc4b6f0926344f04af7a83af91imperiallat=-8.5&lon=-36.95&units=' with block values must be str, not float



```python
plt.scatter(citypy_CSV["Lattitude,"],citypy_CSV["Temperature"], alpha = 0.5)
plt.title ("Temperature vs. Lattitude")
plt.xlabel("Lattitude")
plt.ylabel("Temperature (F)")
plt.savefig("Temperature.png")
plt.show()
```


```python
plt.scatter(citypy_CSV["Lattitude"], citypy_CSV["Cloudiness"], alpha = 0.75)
plt.title("Lattitude vs Cloudiness")
plt.xlabel("Lattitude")
plt.ylabel("Cloudiness")
plt.savefig("Cloudiness.png")
plt.show()
```


```python
plt.scatter(citypy_CSV["Lattitude"], citypy_CSV["Wind Speed"], alpha = 0.85)
plt.title("Lattitude vs Wind Speed")
plt.xlabel("Lattitude")
plt.ylabel("Wind Speed")
plt.savefig("Wind Speed.png")
plt.show()
```
