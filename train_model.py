import pandas as pd
import joblib

from pathlib import Path
from urllib.request import urlopen
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "plotly/datasets/master/diabetes.csv"
)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

def main():
    print("Downloading diabetes dataset...")

    try:
        with urlopen(DATA_URL, timeout=30) as response:
            data = pd.read_csv(BytesIO(response.read()))
    except Exception as error:
        print("Dataset download failed:", error)
        print("Check your internet connection and try again.")
        return

    required_columns = FEATURES + ["Outcome"]

    if not set(required_columns).issubset(data.columns):
        print("The downloaded dataset has unexpected columns.")
        return

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=["Outcome"])
    data = data[data["Outcome"].isin([0, 1])]

    # Zero values in these measurements are treated as missing.
    # Zero is retained for Pregnancies.
    for column in [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]:
        data[column] = data[column].replace(0, float("nan"))

    X = data[FEATURES]
    y = data["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=3,
            ),
        ),
    ])

    print("Training the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTest-set accuracy:")
    print(f"{accuracy_score(y_test, predictions):.2%}")

    print("\nTest-set classification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH,
    )

    print(f"\nModel saved successfully: {MODEL_PATH.name}")
    print("You can now start the Flask website.")

if __name__ == "__main__":
    main()