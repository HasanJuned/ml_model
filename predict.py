import sys
import json
import joblib
import numpy as np

# Load the trained model
model = joblib.load("logreg_model.pkl")

try:
    # Read full stdin stream
    input_json = sys.stdin.read()
    input_data = json.loads(input_json)

    # Ensure correct order of features
    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    features = [input_data[feature] for feature in feature_order]
    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)[0]

    print(json.dumps({"prediction": int(prediction)}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
