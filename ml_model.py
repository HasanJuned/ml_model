# heart_disease_prediction.py

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows with any negative values
    df = df[~(df < 0).any(axis=1)]

    # Detect and remove outliers in specified columns
    def detect_outliers(df, column, low, up):
        return df[(df[column] < low) | (df[column] > up)].index.tolist()

    outliers = []
    outliers += detect_outliers(df, 'age', 10, 100)
    outliers += detect_outliers(df, 'sex', 0, 1)
    outliers += detect_outliers(df, 'bmi', 14, 55)
    outliers += detect_outliers(df, 'cp', 0, 2)
    outliers += detect_outliers(df, 'trestbps', 90, 200)
    outliers += detect_outliers(df, 'chol', 50, 600)
    outliers += detect_outliers(df, 'fbs', 0, 1)
    outliers += detect_outliers(df, 'diabetes', 0, 1)
    outliers += detect_outliers(df, 'maxHR', 60, 200)
    outliers += detect_outliers(df, 'smoker', 0, 2)

    df_cleaned = df.drop(list(set(outliers)))

    return df_cleaned


def preprocess_data(df):
    # Normalize numerical features
    numerical_features = ['age', 'bmi', 'trestbps', 'chol']
    categorical_features = ['sex', 'cp', 'diabetes', 'smoker', 'target']

    scaler = StandardScaler()
    df_num = pd.DataFrame(scaler.fit_transform(df[numerical_features]),
                          columns=numerical_features, index=df.index)

    df_cat = df[categorical_features].copy()

    # One-hot encode categorical variables except target
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = pd.DataFrame(encoder.fit_transform(df_cat.drop(columns=['target'])),
                               columns=encoder.get_feature_names_out(categorical_features[:-1]),
                               index=df.index)

    # Combine all features and target
    df_processed = pd.concat([df_num, cat_encoded, df_cat['target']], axis=1)

    return df_processed, encoder, scaler


def train_models(X_train, y_train):
    models = {
        "SVM": SVC(kernel='linear', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=300),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")

    return models


def evaluate_models(models, X_test, y_test):
    print("\nModel Evaluation:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))


def predict_user_input(svm_model, encoder, scaler):
    # Get user input
    age = float(input("Enter age: "))
    bmi = float(input("Enter bmi: "))
    trestbps = float(input("Enter trestbps: "))
    chol = float(input("Enter chol: "))
    sex = int(input("Enter sex (0 or 1): "))
    cp = int(input("Enter cp (0 to 3): "))
    diabetes = int(input("Enter diabetes (0 or 1): "))
    smoker = int(input("Enter smoker (0 to 2): "))

    # Prepare categorical features as per OneHotEncoder
    cat_input = pd.DataFrame([[sex, cp, diabetes, smoker]], columns=['sex', 'cp', 'diabetes', 'smoker'])
    cat_encoded = encoder.transform(cat_input)

    # Normalize numerical features
    num_input = pd.DataFrame([[age, bmi, trestbps, chol]], columns=['age', 'bmi', 'trestbps', 'chol'])
    num_scaled = scaler.transform(num_input)

    # Combine into one feature vector
    features = pd.DataFrame(pd.np.hstack((num_scaled, cat_encoded)),
                            columns=list(scaler.feature_names_in_) + list(encoder.get_feature_names_out()),
                            index=[0])

    prediction = svm_model.predict(features)

    print(f"Prediction (Heart Disease: 1, No Heart Disease: 0): {prediction[0]}")


def main():
    df_cleaned = load_and_clean_data('3-Senior_Apu_heart.csv')
    df_processed, encoder, scaler = preprocess_data(df_cleaned)

    X = df_processed.drop('target', axis=1)
    y = df_processed['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    # Save SVM model and transformers for later use
    joblib.dump(models['SVM'], 'svm_model.joblib')
    joblib.dump(encoder, 'encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Predict on user input
    predict_user_input(models['SVM'], encoder, scaler)


if __name__ == "__main__":
    main()
