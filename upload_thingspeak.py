import pandas as pd
import requests
import time

API_KEY = "9S0BENPYOQMHWOOD"

data = pd.read_csv("crop2.csv")

print("Dataset loaded:", len(data))

for i in range(len(data)):

    payload = {
        "api_key": API_KEY,
        "field1": data["N"][i],
        "field2": data["P"][i],
        "field3": data["K"][i],
        "field4": data["temperature"][i],
        "field5": data["humidity"][i],
        "field6": data["ph"][i],
        "field7": data["rainfall"][i]
    }

    requests.get("https://api.thingspeak.com/update", params=payload)

    print("Uploaded row:", i)

    time.sleep(15)