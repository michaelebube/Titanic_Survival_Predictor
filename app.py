from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# Attempt to load trained model from a few common names and give helpful errors
def load_model():
    candidates = [
        "titanic_model.joblib",
        "model.joblib",
        "model.pkl",
    ]

    for p in candidates:
        try:
            print(f"Trying to load model from {p}")
            m = joblib.load(p)
            print("Model loaded from", p)
            return m
        except FileNotFoundError:
            continue
        except Exception as e:
            print("Error loading model from", p)
            print(e)
            print(
                "Likely cause: scikit-learn version mismatch between training and runtime."
            )
            print(
                "Options: (1) install the sklearn version used for training, e.g. `pip install scikit-learn==1.6.1`\n         (2) retrain the model in this environment and save as 'model.joblib'."
            )
            raise


model = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Pclass": int(request.form.get("Pclass", 3)),
        "Sex": request.form.get("Sex", "male"),
        "Age": float(request.form.get("Age")) if request.form.get("Age") else None,
        "SibSp": int(request.form.get("SibSp", 0)),
        "Parch": int(request.form.get("Parch", 0)),
        "Fare": (
            float(request.form.get("Fare", 0.0)) if request.form.get("Fare") else 0.0
        ),
        "Embarked": request.form.get("Embarked", "S"),
    }

    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prediction_text = "Survived" if int(pred) == 1 else "Not Survived"

    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
