from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('like_predictor.pkl')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['word_count'],
        data['char_count'],
        data['has_media'],
        data['hour'],
        data['sentiment'],
        data['company_encoded'],
        data['username_encoded'],
        data['day_of_week'],
    ]).reshape(1, -1)

    prediction_log = model.predict(features)[0]
    prediction = round(np.expm1(prediction_log))
    return jsonify({'predicted_likes': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
