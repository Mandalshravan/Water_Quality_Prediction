from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load pipeline (imputer + scaler + model together)
model = joblib.load("svm_pipeline.pkl")

# Feature names must match training data column order exactly
FEATURE_NAMES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                  'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_waterQ', methods=['GET', 'POST'])
def predict_waterQ():
    ph = float(request.form["ph"])
    hardness = float(request.form["hardness"])
    solids = float(request.form["solids"])
    chloramines = float(request.form["chloramines"])
    sulfate = float(request.form["sulfate"])
    conductivity = float(request.form["conductivity"])
    organicCarbon = float(request.form["organicCarbon"])
    trihalomethanes = float(request.form["trihalomethanes"])
    turbidity = float(request.form["turbidity"])

    # Use DataFrame with correct column names so pipeline works properly
    input_df = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                               conductivity, organicCarbon, trihalomethanes, turbidity]],
                             columns=FEATURE_NAMES)

    prediction = model.predict(input_df)[0]
    prediction_new = 'Safe' if prediction == 1 else 'Unsafe'

    return render_template('result.html', prediction=prediction_new)

if __name__ == '__main__':
    app.run(debug=True)
