import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings 
warnings.filterwarnings("ignore")

data = pd.read_csv("crop.csv")

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# --------------------------
# Encode labels
# --------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# --------------------------
# Train Test Split FIRST (before scaling!)
# ⚠️ Previous code scaled ALL data before splitting = data leakage
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Feature Scaling (fit ONLY on train, transform both)
# --------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on train only
X_test  = scaler.transform(X_test)        # transform test using train's stats

# --------------------------
# Train Model - Stricter RF to reduce overfitting
# --------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,              # reduced from 8
    min_samples_split=10,     # increased to prevent small splits
    min_samples_leaf=5,       # increased leaf size
    max_features='sqrt',      # limit features per split (default, but explicit)
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained successfully\n")

# --------------------------
# Overfitting Diagnosis
# --------------------------
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy  = accuracy_score(y_test,  model.predict(X_test))

print("=" * 45)
print("🔍 Overfitting Diagnosis")
print("=" * 45)
print(f"  Train Accuracy : {train_accuracy * 100:.2f}%")
print(f"  Test  Accuracy : {test_accuracy  * 100:.2f}%")
gap = (train_accuracy - test_accuracy) * 100
print(f"  Gap            : {gap:.2f}%")

if gap < 2:
    print("  ✅ No significant overfitting detected")
elif gap < 5:
    print("  ⚠️  Slight overfitting — consider tightening constraints")
else:
    print("  ❌ Overfitting detected — model needs more regularization")

# --------------------------
# Classification Report
# --------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print(f"✅ Test Accuracy: {round(test_accuracy * 100, 2)}%")

# --------------------------
# Stratified Cross Validation (more reliable than simple CV)
# --------------------------
X_all_scaled = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_all_scaled, y_all, cv=cv, scoring='accuracy')

print("\n📊 Cross Validation Results (5-fold Stratified):")
print(f"  Fold scores : {[round(s * 100, 2) for s in scores]}")
print(f"  Mean        : {scores.mean() * 100:.2f}%")
print(f"  Std Dev     : {scores.std() * 100:.2f}%")

if scores.std() < 0.02:
    print("  ✅ Model is stable across folds (low variance)")
else:
    print("  ⚠️  High variance across folds — possible instability")

# --------------------------
# Feature Importance
# --------------------------
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n📈 Feature Importances:")
for i in sorted_idx:
    print(f"  {feature_names[i]:<12}: {importances[i] * 100:.2f}%")

# --------------------------
# Predict Crop from Sample Sensor Data
# --------------------------
sample = pd.DataFrame(
    [[90, 40, 40, 25, 80, 6.5, 200]],
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
)

sample_scaled = scaler.transform(sample)     # use the SAME scaler fitted on train
prediction    = model.predict(sample_scaled)
probabilities = model.predict_proba(sample_scaled)[0]
crop          = encoder.inverse_transform(prediction)

print(f"\n🌱 Predicted Crop : {crop[0]}")
print(f"   Confidence     : {max(probabilities) * 100:.1f}%")

# Top 3 crop suggestions
top3_idx  = np.argsort(probabilities)[::-1][:3]
top3_crops = encoder.inverse_transform(top3_idx)
print("\n   Top 3 Suggestions:")
for i, (c, p) in enumerate(zip(top3_crops, probabilities[top3_idx]), 1):
    print(f"   {i}. {c:<15} ({p * 100:.1f}%)")

# --------------------------
# Read Data from ThingSpeak & Predict
# --------------------------
CHANNEL_ID   = "3295651"
READ_API_KEY = "0RD8TEPFFJYRSYTI"

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_API_KEY}"

try:
    thingspeak_data = pd.read_csv(url)
    print("\n📡 Data from ThingSpeak:")
    print(thingspeak_data.head())

    # Map ThingSpeak fields to model features
    # Adjust field1-field7 mapping based on your actual channel setup
    field_map = {
        'field1': 'N',
        'field2': 'P',
        'field3': 'K',
        'field4': 'temperature',
        'field5': 'humidity',
        'field6': 'ph',
        'field7': 'rainfall'
    }

    live_data = thingspeak_data[list(field_map.keys())].dropna().tail(1)

    if not live_data.empty:
        live_data = live_data.rename(columns=field_map)
        live_scaled    = scaler.transform(live_data)
        live_pred      = model.predict(live_scaled)
        live_crop      = encoder.inverse_transform(live_pred)
        print(f"\n🌾 Live Sensor Prediction: {live_crop[0]}")
    else:
        print("⚠️  No valid live data found in ThingSpeak feed")

except Exception as e:
    print(f"\n⚠️  Could not fetch ThingSpeak data: {e}")
