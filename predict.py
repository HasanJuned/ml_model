import sys
import json
import joblib
import numpy as np

# Load the model
model = joblib.load("logreg_model.pkl")

try:
    # Read JSON from stdin
    input_data = json.load(sys.stdin)

    # Feature order (must match training)
    features = [
        input_data["age"], input_data["sex"], input_data["cp"],
        input_data["trestbps"], input_data["chol"], input_data["fbs"],
        input_data["restecg"], input_data["thalach"], input_data["exang"],
        input_data["oldpeak"], input_data["slope"], input_data["ca"],
        input_data["thal"]
    ]

    features = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    # Output result
    print(json.dumps({"prediction": int(prediction[0])}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
