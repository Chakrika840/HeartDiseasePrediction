import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):

    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    df.drop(["id", "dataset"], axis=1, inplace=True)

    # Convert TRUE/FALSE
    df["fbs"] = df["fbs"].map({"TRUE": 1, "FALSE": 0})
    df["exang"] = df["exang"].map({"TRUE": 1, "FALSE": 0})

    # Convert target to binary
    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    # One-hot encoding
    X = df.drop("num", axis=1)
    y = df["num"]

    X = pd.get_dummies(X, drop_first=True)

    # Safety fill
    X = X.fillna(0)

    columns = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, columns