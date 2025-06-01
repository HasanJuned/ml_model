import sys
import joblib
import pandas as pd
import numpy as np

# Load your model, encoder, scaler (make sure these files exist in same dir)
svm_model = joblib.load('svm_model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

def predict_heart_disease(age, bmi, trestbps, chol, sex, cp, diabetes, smoker):
    # Prepare categorical features as per OneHotEncoder
    cat_input = pd.DataFrame([[sex, cp, diabetes, smoker]], columns=['sex', 'cp', 'diabetes', 'smoker'])
    cat_encoded = encoder.transform(cat_input)

    # Normalize numerical features
    num_input = pd.DataFrame([[age, bmi, trestbps, chol]], columns=['age', 'bmi', 'trestbps', 'chol'])
    num_scaled = scaler.transform(num_input)

    # Combine features
    features = np.hstack((num_scaled, cat_encoded))

    prediction = svm_model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    # Expect 8 arguments: age bmi trestbps chol sex cp diabetes smoker
    if len(sys.argv) != 9:
        print("Error: Expected 8 inputs")
        sys.exit(1)
    inputs = list(map(float, sys.argv[1:5])) + list(map(int, sys.argv[5:9]))
    result = predict_heart_disease(*inputs)
    print(result)
