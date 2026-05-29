
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ✅ Load data
history = pd.read_excel("passenger_purchase_history_200.xlsx", engine="openpyxl")
trips = pd.read_excel("trips_with_flight_1000.xlsx", engine="openpyxl")

# ✅ Merge trip info
data = history.merge(trips, on=["Departure", "Arrival"], how="left")

# ✅ Fill missing values
data.fillna({
    "ConnectionRiskScore": 0.5,
    "DisruptionProbability": 0.4,
    "TimeCriticality": "Medium"
}, inplace=True)

# ✅ Encode categories
le_loyalty = LabelEncoder()
le_tier = LabelEncoder()
le_time = LabelEncoder()

data["loyalty_enc"] = le_loyalty.fit_transform(data["Loyalty_Tier"])
data["tier_enc"] = le_tier.fit_transform(data["ServiceTier"])
data["time_enc"] = le_time.fit_transform(data["TimeCriticality"])

# ✅ Features
X = data[
    ["ConnectionRiskScore", "DisruptionProbability", "time_enc", "loyalty_enc", "Usage_Count"]
]

# ✅ Target
y = data["tier_enc"]

# ✅ Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# ✅ Save model + encoders
joblib.dump(model, "propensity_model.pkl")
joblib.dump(le_loyalty, "le_loyalty.pkl")
joblib.dump(le_tier, "le_tier.pkl")
joblib.dump(le_time, "le_time.pkl")

print("✅ Model Training Completed!")

