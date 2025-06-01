import sys
import json
import joblib
import numpy as np
import traceback

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Define expected features
expected_features = [
    "age", "sex", "cp",
    "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca",
    "thal"
]

# Optional fallback input (for testing without stdin)
fallback_input = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

# Load model
try:
    model = joblib.load("logreg_model.pkl")
except Exception as e:
    eprint("Error loading model:", e)
    sys.exit(1)

# Read input
try:
    if sys.stdin.isatty():
        # No input provided, use fallback
        input_data = fallback_input
        eprint("No input detected. Using fallback input.")
    else:
        input_data = json.load(sys.stdin)
except Exception as e:
    eprint("Error reading input JSON:", e)
    print(json.dumps({"error": "Invalid or missing JSON input"}), flush=True)
    sys.exit(1)

# Process input and make prediction
try:
    features = []
    for feat in expected_features:
        if feat not in input_data:
            raise ValueError(f"Missing feature: {feat}")
        features.append(float(input_data[feat]))

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    print(json.dumps({"prediction": int(prediction[0])}), flush=True)

except Exception as e:
    eprint("Prediction error:", e)
    eprint(traceback.format_exc())
    print(json.dumps({"error": str(e)}), flush=True)
