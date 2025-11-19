#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
import requests
import time
import pymongo
from datetime import datetime, timezone

load_dotenv()


# In[2]:


OWM_KEY = os.environ["OpenWeatherMaps_Api_Key"]


# In[30]:


#def fetch_weather(lat, lon):
#    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
#    r = requests.get(url)
#    return r.json()


#fetch_weather(19, 73)


# In[ ]:





# In[3]:


MONGO_URI = "mongodb://localhost:27017"

client = pymongo.MongoClient(MONGO_URI)
db = client.weather
raw = db.raw_observations

LOCATIONS = [
    {"id": f"sg-{i+1:02d}", "lat": lat, "lon": lon}
    for i, (lat, lon) in enumerate([
        (1.290270, 103.851959), (1.352083, 103.819836), (1.280095, 103.850949),
        (1.300000, 103.800000), (1.320000, 103.830000), (1.340000, 103.750000),
        (1.370000, 103.800000), (1.380000, 103.890000), (1.360000, 103.950000),
        (1.310000, 103.940000), (1.280000, 103.870000), (1.330000, 103.880000),
        (1.360000, 103.870000), (1.300000, 103.820000), (1.320000, 103.840000),
        (1.350000, 103.860000), (1.370000, 103.830000), (1.390000, 103.780000),
        (1.400000, 103.900000), (1.410000, 103.810000), (1.420000, 103.830000),
        #(1.430000, 103.850000), (1.440000, 103.870000), (1.450000, 103.890000),
        #(1.460000, 103.910000), (1.470000, 103.930000), (1.280000, 103.830000),
        #(1.285000, 103.805000), (1.290000, 103.780000), (1.295000, 103.755000),
        #(1.300000, 103.730000), (1.305000, 103.705000), (1.310000, 103.680000),
        #(1.315000, 103.655000), (1.320000, 103.630000), (1.325000, 103.605000),
        #(1.330000, 103.580000), (1.335000, 103.555000), (1.340000, 103.530000),
        #(1.345000, 103.505000), (1.350000, 103.480000), (1.355000, 103.455000),
        #(1.360000, 103.430000), (1.365000, 103.405000), (1.370000, 103.380000),
        #(1.375000, 103.355000), (1.380000, 103.330000), (1.385000, 103.305000),
        #(1.390000, 103.280000), (1.395000, 103.255000), (1.400000, 103.230000),
        #(1.405000, 103.205000), (1.410000, 103.180000), (1.415000, 103.155000),
        #(1.420000, 103.130000), (1.425000, 103.105000), (1.430000, 103.080000),
        #(1.435000, 103.055000), (1.440000, 103.030000), (1.445000, 103.005000),
        #(1.450000, 102.980000), (1.455000, 102.955000),
    ])
]

def fetch_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    while True:
        start = time.time()
        for loc in LOCATIONS:
            try:
                data = fetch_weather(loc["lat"], loc["lon"])
                doc = {
                    "location_id": loc["id"],
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "timestamp": datetime.now(timezone.utc),
                    "payload": data,
                }
                raw.insert_one(doc)
                print(f"[{loc['id']}] OK at {doc['timestamp']}")
            except Exception as e:
                print(f"[{loc['id']}] ERROR: {e}")
        # keep 60s spacing between batches
        time.sleep(max(0, 60 - (time.time() - start)))

if __name__ == "__main__":
    main()

