import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_dataset():
    dataset = fetch_ucirepo(id=547)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    y = y.astype(str).str.strip().str.lower()
    y = y.replace({
        "fire": 1,
        "not fire": 0,
        "notfire": 0
    })

    y = pd.to_numeric(y, errors="coerce")
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    X.columns = [str(col).strip().lower().replace(" ", "_") for col in X.columns]

    return X, y


def main():
    model = joblib.load("wildfire_best_model.joblib")
    X, y = load_dataset()

    predictions = model.predict(X)

    print("Full Dataset Accuracy:", round(accuracy_score(y, predictions), 4))
    print("\nClassification Report:")
    print(classification_report(y, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))


if __name__ == "__main__":
    main()
