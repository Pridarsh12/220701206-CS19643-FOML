# %%
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
with open("salary_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    label_encoders = saved["label_encoders"]
    feature_columns = saved["feature_columns"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = []
        for col in feature_columns:
            val = request.form[col]
            if col in label_encoders:
                val = label_encoders[col].transform([val])[0]
            else:
                val = float(val)
            user_input.append(val)

        input_df = pd.DataFrame([user_input], columns=feature_columns)
        prediction = model.predict(input_df).item()
        prediction = f"\u20B9{int(prediction):,} per year"

    # Options for form
    options = {
    col: label_encoders[col].classes_.tolist() if col in label_encoders else None
    for col in feature_columns
}

    return render_template("index.html", options=options, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)




