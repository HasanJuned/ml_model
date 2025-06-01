import sys
import json
import joblib
import numpy as np

model = joblib.load("logreg_model.pkl")

try:
    input_data = json.load(sys.stdin)

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
