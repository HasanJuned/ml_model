import sys
import json
import joblib
import numpy as np

# Load the trained model
model = joblib.load("logreg_model.pkl")

# Read JSON input from command line argument
input_data = json.loads(sys.argv[1])  # {"features": [x, y, z, ...]}

# Convert input to numpy array
features = np.array(input_data["features"]).reshape(1, -1)

# Get prediction
prediction = model.predict(features)

# Return result as JSON
print(json.dumps({"prediction": prediction.tolist()}))
