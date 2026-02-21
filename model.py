from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier


def build_model():

    base_models = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("adb", AdaBoostClassifier(n_estimators=100))
    ]

    final_estimator = LogisticRegression()

    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator
    )

    return stack_model