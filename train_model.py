import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

df = df.drop(columns=["UDI", "Product ID"])

df["Failure"] = (
    df["Machine failure"] | df["TWF"] | df["HDF"] | df["PWF"] | df["OSF"] | df["RNF"]
)

df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

X = df[["Type", "Air temperature [K]", "Process temperature [K]", 
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]]
y = df["Failure"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved successfully (NO TensorFlow)")