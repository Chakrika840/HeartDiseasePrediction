import joblib
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import load_and_preprocess_data
from model import build_model
def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n====================================")
    print(f"{name}")
    print("====================================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()

    return model


def train():
    X_train, X_test, y_train, y_test, scaler, columns = load_and_preprocess_data("data/heart.csv")

    lr = LogisticRegression(max_iter=1000)
    evaluate_model(lr, "Logistic Regression", X_train, X_test, y_train, y_test)

    rf = RandomForestClassifier(n_estimators=100)
    evaluate_model(rf, "Random Forest", X_train, X_test, y_train, y_test)

    adb = AdaBoostClassifier(n_estimators=100)
    evaluate_model(adb, "AdaBoost", X_train, X_test, y_train, y_test)

    stack_model = build_model()
    stack_model = evaluate_model(stack_model, "Stacking Model", X_train, X_test, y_train, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(stack_model, "models/heart_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(columns, "models/columns.pkl")


if __name__ == "__main__":
    train()