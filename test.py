import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


data = pd.read_csv("crop.csv")

X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']


encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("✅ Model trained successfully")

y_pred = model.predict(X_test)

print("\n📊 Classification Report\n")
print(classification_report(y_test, y_pred))

sample = pd.DataFrame([[90,40,40,25,80,6.5,200]],
columns=['N','P','K','temperature','humidity','ph','rainfall'])

prediction = model.predict(sample)

crop = encoder.inverse_transform(prediction)

print("\n🌱 Predicted Crop:", crop[0])


CHANNEL_ID = "3295651"
READ_API_KEY = "0RD8TEPFFJYRSYTI"

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_API_KEY}"

thingspeak_data = pd.read_csv(url)

print("\n📡 Data from ThingSpeak:")
print(thingspeak_data.head())
