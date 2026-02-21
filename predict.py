import joblib
import pandas as pd

model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


def predict_heart(data_dict):
    df = pd.DataFrame([data_dict])

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        return "High Risk of Heart Disease"
    else:
        return "Low Risk of Heart Disease"