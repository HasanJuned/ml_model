import sys
import json
import joblib
import pandas as pd

# Load saved objects
svm_model = joblib.load('svm_model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

def predict(input_data):
    # Extract fields
    age = float(input_data['age'])
    bmi = float(input_data['bmi'])
    trestbps = float(input_data['trestbps'])
    chol = float(input_data['chol'])
    sex = int(input_data['sex'])
    cp = int(input_data['cp'])
    diabetes = int(input_data['diabetes'])
    smoker = int(input_data['smoker'])

    cat_input = pd.DataFrame([[sex, cp, diabetes, smoker]], columns=['sex', 'cp', 'diabetes', 'smoker'])
    cat_encoded = encoder.transform(cat_input)

    num_input = pd.DataFrame([[age, bmi, trestbps, chol]], columns=['age', 'bmi', 'trestbps', 'chol'])
    num_scaled = scaler.transform(num_input)

    features = pd.DataFrame(pd.np.hstack((num_scaled, cat_encoded)),
                            columns=list(scaler.feature_names_in_) + list(encoder.get_feature_names_out()))

    prediction = svm_model.predict(features)[0]

    return prediction

def main():
    input_str = sys.stdin.read()
    input_json = json.loads(input_str)
    pred = predict(input_json)
    print(json.dumps({"prediction": int(pred)}))

if __name__ == '__main__':
    main()
