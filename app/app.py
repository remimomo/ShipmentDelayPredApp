from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from math import radians, cos, sin, sqrt, atan2
import requests

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Your OpenWeatherMap API key (replace with your actual API key)
WEATHER_API_KEY = "d41aff3e76dd473606fee6315c755280"

# Function to calculate distance between two geographic points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lat2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to fetch current weather data for a given location
def get_weather(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Parse the form data
        form_data = request.form.to_dict()
        data = {
            "SHIPMENT_NUMBER": form_data['shipment_number'],
            "VEHICLE_SIZE": form_data['vehicle_size'],
            "VEHICLE_BUILD_UP": form_data['vehicle_build_up'],
            "FIRST_COLLECTION_LATITUDE": float(form_data['first_collection_latitude']),
            "FIRST_COLLECTION_LONGITUDE": float(form_data['first_collection_longitude']),
            "LAST_DELIVERY_LATITUDE": float(form_data['last_delivery_latitude']),
            "LAST_DELIVERY_LONGITUDE": float(form_data['last_delivery_longitude']),
            "LAST_DELIVERY_SCHEDULE_EARLIEST": form_data['last_delivery_schedule_earliest'],
            "LAST_DELIVERY_SCHEDULE_LATEST": form_data['last_delivery_schedule_latest'],
            "SHIPPER_ID": form_data['shipper_id'],
            "CARRIER_ID": form_data['carrier_id']
        }

        df = pd.DataFrame([data])

        # Feature Engineering
        df['DELIVERY_WINDOW'] = (pd.to_datetime(df['LAST_DELIVERY_SCHEDULE_LATEST']) - pd.to_datetime(df['LAST_DELIVERY_SCHEDULE_EARLIEST'])).dt.total_seconds() / 3600
        df['DISTANCE'] = df.apply(lambda row: haversine(row['FIRST_COLLECTION_LATITUDE'], row['FIRST_COLLECTION_LONGITUDE'], row['LAST_DELIVERY_LATITUDE'], row['LAST_DELIVERY_LONGITUDE']), axis=1)
        
        # Fetch weather data and add it as a feature
        weather = get_weather(df.loc[0, 'LAST_DELIVERY_LATITUDE'], df.loc[0, 'LAST_DELIVERY_LONGITUDE'], WEATHER_API_KEY)
        df['DELIVERY_WEATHER'] = weather['main']['temp'] if weather else None

        # Prepare the feature set for prediction
        features = ['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'DELIVERY_WINDOW', 'DISTANCE', 'DELIVERY_WEATHER', 'SHIPPER_ID', 'CARRIER_ID']
        X_new = pd.get_dummies(df[features], columns=['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'SHIPPER_ID', 'CARRIER_ID'])
        
        # Make prediction
        prediction = model.predict(X_new)[0]
        result = "On Time" if prediction else "Delayed"

        return render_template('index.html', result=result)

    return render_template('index.html') 

if __name__ == '__main__':
     # Run the Flask app
     app.run(debug=True)