import sys
import json
import joblib
import numpy as np

# Load the trained model
model = joblib.load("logreg_model.pkl")

try:
    # Get input JSON as dictionary
    input_data = json.loads(sys.argv[1])  # expects object, not array

    # Maintain feature order used in training
    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Convert to ordered feature list
    features = [input_data[feature] for feature in feature_order]
    features_array = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(features_array)[0]

    # Return JSON output
    print(json.dumps({"prediction": int(prediction)}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
