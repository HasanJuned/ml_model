import sys
import json
import joblib
import numpy as np
import traceback

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

try:
    # Load the model
    model = joblib.load("logreg_model.pkl")
except Exception as e:
    eprint("Error loading model:", e)
    sys.exit(1)

try:
    # Read JSON from stdin
    input_data = json.load(sys.stdin)
except Exception as e:
    eprint("Error reading input JSON:", e)
    sys.exit(1)

# List of expected features in the correct order
expected_features = [
    "age", "sex", "cp",
    "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca",
    "thal"
]

features = []
try:
    for feat in expected_features:
        if feat not in input_data:
            raise ValueError(f"Missing feature: {feat}")
        # Convert each feature to float or int depending on your model requirement
        # Here, converting all to float to be safe
        features.append(float(input_data[feat]))

    features = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    # Output JSON result
    print(json.dumps({"prediction": int(prediction[0])}), flush=True)

except Exception as e:
    # Print error JSON to stdout (as required)
    # You can also print traceback to stderr for debugging
    eprint("Exception during prediction:", e)
    eprint(traceback.format_exc())
    print(json.dumps({"error": str(e)}), flush=True)
