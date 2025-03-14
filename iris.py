from flask import Flask, render_template, request
import pickle
import numpy as np

# Load saved components
model = pickle.load(open('iri.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        data = []
        features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
        for feature in features:
            value = request.form[feature]
            if feature in ['Gender', 'Blood Pressure', 'Cholesterol Level']:  # Categorical fields
                value = label_encoders[feature].transform([value])[0]
            else:
                value = int(value)  # Numeric fields
            data.append(value)

        # Scale the input
        arr = np.array([data])
        arr_scaled = scaler.transform(arr)

        # Predict the disease
        pred = model.predict(arr_scaled)
        pred_prob = model.predict_proba(arr_scaled)

        # Get the top 3 predicted diseases
        top_3_indices = np.argsort(pred_prob[0])[-3:][::-1]
        top_3_diseases = label_encoders['Disease'].inverse_transform(top_3_indices)

        most_predicted_disease = label_encoders['Disease'].inverse_transform([pred[0]])[0]

        return render_template('after.html', most_predicted=most_predicted_disease, top_3_diseases=top_3_diseases)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)