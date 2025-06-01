import json
import joblib
import numpy as np

# Load model
model = joblib.load("logreg_model.pkl")

# Hardcoded input
features = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Predict
prediction = model.predict(features)

# Output JSON
print(json.dumps({"prediction": int(prediction[0])}), flush=True)
