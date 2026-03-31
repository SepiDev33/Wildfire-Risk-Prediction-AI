import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


def main():
    X, y = load_dataset()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200)
    }

    best_pipeline = None
    best_name = None
    best_score = -1

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        print(f"\n=== {name} ===")
        print("Accuracy:", round(acc, 4))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        if acc > best_score:
            best_score = acc
            best_pipeline = pipeline
            best_name = name

    print(f"\nBest model: {best_name} with accuracy = {round(best_score, 4)}")

    joblib.dump(best_pipeline, "wildfire_best_model.joblib")
    print("Saved model to wildfire_best_model.joblib")


if __name__ == "__main__":
    main()
