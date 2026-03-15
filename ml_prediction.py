import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("crop_data.csv")

X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train,y_train)

print("Model trained successfully")

prediction = model.predict([[90,40,40,25,80,6.5,200]])

print("Predicted Crop:", encoder.inverse_transform(prediction))

import pandas as pd

CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API = "YOUR_READ_API_KEY"

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_API}"

data = pd.read_csv(url)

print(data.head())