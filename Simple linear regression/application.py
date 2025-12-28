from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

regression_path = os.path.join(BASE_DIR, 'regression.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler-data.pkl')

regression_model = pickle.load(open(regression_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        weight = float(request.form['Weight'])
        new_data = np.array([[weight]])
        new_data_scaled = scaler.transform(new_data)
        prediction = regression_model.predict(new_data_scaled)[0]

    return render_template('index.html', results=prediction)

if __name__ == '__main__':
    app.run(debug=True)
