import gradio as gr
from src.predict import predict_heart # type: ignore


def user_input(
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
):
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "TRUE" else 0,
        "restecg": restecg,
        "thalch": thalach,
        "exang": 1 if exang == "TRUE" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    return predict_heart(data)


interface = gr.Interface(
    fn=user_input,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Male", "Female"], label="Sex"),
        gr.Dropdown(["typical angina","atypical angina","non-anginal","asymptomatic"], label="Chest Pain"),
        gr.Number(label="Resting BP"),
        gr.Number(label="Cholesterol"),
        gr.Dropdown(["TRUE","FALSE"], label="Fasting Blood Sugar"),
        gr.Dropdown(["normal","lv hypertrophy"], label="Rest ECG"),
        gr.Number(label="Max Heart Rate"),
        gr.Dropdown(["TRUE","FALSE"], label="Exercise Angina"),
        gr.Number(label="Oldpeak"),
        gr.Dropdown(["upsloping","flat","downsloping"], label="Slope"),
        gr.Number(label="Major Vessels"),
        gr.Dropdown(["normal","fixed defect","reversable defect"], label="Thal")
    ],
    outputs="text",
    title="Heart Disease Prediction System",
    description="Stacking Ensemble: Logistic Regression + Random Forest + AdaBoost"
)

if __name__ == "__main__":
    interface.launch()