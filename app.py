from flask import Flask, render_template, request, jsonify
import requests
import pickle
import pandas as pd

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Function to get real-time weather data
def get_weather(lat, lon):
    api_key = "d41aff3e76dd473606fee6315c755280"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(weather_url)
    weather_data = response.json()
    if weather_data.get('cod') == 200:
        weather_condition = weather_data['weather'][0]['main']
        temperature = weather_data['main']['temp']
        return weather_condition, temperature
    else:
        return None, None

# Function to get real-time traffic data
def get_traffic(lat, lon):
    api_key = "your_google_maps_api_key"
    traffic_url = f"https://maps.googleapis.com/maps/api/traffic/json?location={lat},{lon}&key={api_key}"
    response = requests.get(traffic_url)
    traffic_data = response.json()
    # Process traffic data as needed
    # Example return value: traffic_condition = 'Heavy', 'Moderate', 'Light'
    traffic_condition = "Moderate"  # Placeholder
    return traffic_condition

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    vehicle_size = request.form['vehicle_size']
    vehicle_build_up = request.form['vehicle_build_up']
    first_collection_time = request.form['first_collection_time']
    last_delivery_time = request.form['last_delivery_time']
    destination_lat = request.form['destination_lat']
    destination_lon = request.form['destination_lon']

    # Fetch real-time weather and traffic data for the destination
    weather_condition, temperature = get_weather(destination_lat, destination_lon)
    traffic_condition = get_traffic(destination_lat, destination_lon)

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'VEHICLE_SIZE': [vehicle_size],
        'VEHICLE_BUILD_UP': [vehicle_build_up],
        'FIRST_COLLECTION_SCHEDULE_EARLIEST': [first_collection_time],
        'LAST_DELIVERY_SCHEDULE_LATEST': [last_delivery_time],
        'WEATHER_CONDITION': [weather_condition],
        'TEMPERATURE': [temperature],
        'TRAFFIC_CONDITION': [traffic_condition]
    })

    # Convert categorical features to the same format as the training data
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure the input data has the same columns as the training data
    input_data = input_data.reindex(columns=model.feature_importances_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)

    # Interpret the result
    result = 'Delayed' if prediction == 0 else 'On Time'
    
    # Return the result
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
