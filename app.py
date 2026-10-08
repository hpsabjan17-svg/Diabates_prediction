from flask import Flask, render_template, request
from pathlib import Path
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

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

# The app starts only after the model has been trained.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]


@app.route("/")
def home():
    return render_template(
    "index.html",
    form_data={},
    error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {}

        limits = {
            "Pregnancies": (0, 30),
            "Glucose": (1, 500),
            "BloodPressure": (1, 300),
            "SkinThickness": (0, 150),
            "Insulin": (0, 3000),
            "BMI": (1, 100),
            "DiabetesPedigreeFunction": (0, 10),
            "Age": (1, 120),
        }

        for feature in FEATURES:
            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                raise ValueError(
                    f"Please enter a value for {feature}."
                )

            value = float(raw_value)

            if not pd.notna(value):
                raise ValueError(
                    f"Enter a valid number for {feature}."
                )

            minimum, maximum = limits[feature]

            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{feature} must be between "
                    f"{minimum} and {maximum}."
                )

            values[feature] = value

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES,
        )

        prediction = int(model.predict(input_data)[0])

        # This is a model score, not a clinically validated risk.
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
        score = float(probabilities[positive_index]) * 100

        if prediction == 1:
            result_title = "Higher likelihood indicated"
            result_message = (
                "The model classified this input in its "
                "diabetes-positive category. This is not a diagnosis."
            )
            result_type = "warning"
        else:
            result_title = "Lower likelihood indicated"
            result_message = (
                "The model classified this input in its "
                "diabetes-negative category. This does not rule "
                "out diabetes."
            )
            result_type = "success"

        return render_template(
            "result.html",
            result_title=result_title,
            result_message=result_message,
            result_type=result_type,
            score=round(score, 1),
            values=values,
        )

    except (ValueError, TypeError, OverflowError) as error:
        return render_template(
            "index.html",
            error=str(error),
            form_data=request.form,
        ), 400


if __name__ == "__main__":
    app.run(debug=True)