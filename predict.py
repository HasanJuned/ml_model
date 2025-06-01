import sys
import json
import joblib
import numpy as np

# Load model
model = joblib.load("logreg_model.pkl")

try:
    # Read and parse input
    input_data = json.loads(sys.argv[1])  # {"features": [...]}
    features = np.array(input_data["features"]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    # Output as JSON
    print(json.dumps({"prediction": int(prediction[0])}))
except Exception as e:
    # On error, return JSON-formatted error string to avoid breaking Node.js
    print(json.dumps({"error": str(e)}))
