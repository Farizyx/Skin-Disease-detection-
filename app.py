from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained Logistic Regression model
model = pickle.load(open('lrmodel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)
        classes = {
            1: 'Psoriasis',
            2: 'Seborrheic Dermatitis',
            3: 'Lichen Planus',
            4: 'Pityriasis Rosea',
            5: 'Chronic Dermatitis',
            6: 'Pityriasis Rubra Pilaris'
}
        
        predicted_class = classes[int(prediction[0])]
        
        return render_template('index.html', prediction_text=f'Predicted Skin Disease: {predicted_class}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

