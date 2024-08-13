#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:25:44 2024

@author: remimomo
"""
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
import joblib

# Load the datasets
shipment_bookings = pd.read_csv('/Users/remimomo/Documents/shipment_delay_prediction_v1/data/Shipment_bookings.csv')
gps_data = pd.read_csv('/Users/remimomo/Documents/shipment_delay_prediction_v1/data/GPS_data.csv')
new_bookings = pd.read_csv('/Users/remimomo/Documents/shipment_delay_prediction_v1/data/New_bookings.csv')

# Haversine function to calculate the distance between two latitude-longitude points
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, sqrt, atan2

    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    # Haversine formula to calculate the distance
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

# Convert relevant columns to datetime format
# This is necessary to perform date-based operations accurately
shipment_bookings['FIRST_COLLECTION_SCHEDULE_EARLIEST'] = pd.to_datetime(shipment_bookings['FIRST_COLLECTION_SCHEDULE_EARLIEST'])
shipment_bookings['FIRST_COLLECTION_SCHEDULE_LATEST'] = pd.to_datetime(shipment_bookings['FIRST_COLLECTION_SCHEDULE_LATEST'])
shipment_bookings['LAST_DELIVERY_SCHEDULE_EARLIEST'] = pd.to_datetime(shipment_bookings['LAST_DELIVERY_SCHEDULE_EARLIEST'])
shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'])
gps_data['RECORD_TIMESTAMP'] = pd.to_datetime(gps_data['RECORD_TIMESTAMP'])

# Filter shipments within the specified time period (October 1, 2023 - December 31, 2023)
start_date = pd.Timestamp("2023-10-1", tz="UTC").replace(microsecond=0)
end_date = pd.Timestamp("2023-12-31", tz="UTC").replace(microsecond=0)
#end_date = start_date - pd.Timedelta(days=90)
filtered_shipments = shipment_bookings[
    (shipment_bookings['LAST_DELIVERY_SCHEDULE_EARLIEST'] >= start_date) &
    (shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] <= end_date)
]

# Merge GPS data with filtered shipments on SHIPMENT_NUMBER
merged_data = pd.merge(filtered_shipments, gps_data, on='SHIPMENT_NUMBER', how='left')

# Determine the actual delivery times by finding the last GPS record for each shipment
actual_delivery_times = merged_data.groupby('SHIPMENT_NUMBER')['RECORD_TIMESTAMP'].max().reset_index()

# Merge actual delivery times back with the filtered shipments
delivery_analysis = pd.merge(filtered_shipments, actual_delivery_times, on='SHIPMENT_NUMBER', how='left')
delivery_analysis.rename(columns={'RECORD_TIMESTAMP': 'ACTUAL_DELIVERY_TIME'}, inplace=True)

# Calculate the on-time delivery status
delivery_analysis['ON_TIME'] = delivery_analysis['ACTUAL_DELIVERY_TIME'] <= (delivery_analysis['LAST_DELIVERY_SCHEDULE_LATEST'] + timedelta(minutes=30))

# Feature Engineering: Calculate delivery window duration and distance
delivery_analysis['DELIVERY_WINDOW'] = (delivery_analysis['LAST_DELIVERY_SCHEDULE_LATEST'] - delivery_analysis['LAST_DELIVERY_SCHEDULE_EARLIEST']).dt.total_seconds() / 3600
delivery_analysis['DISTANCE'] = delivery_analysis.apply(lambda row: haversine(row['FIRST_COLLECTION_LATITUDE'], row['FIRST_COLLECTION_LONGITUDE'], row['LAST_DELIVERY_LATITUDE'], row['LAST_DELIVERY_LONGITUDE']), axis=1)

# Prepare the feature set and target variable
features = ['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'DELIVERY_WINDOW', 'DISTANCE', 'SHIPMENT_NUMBER', 'CARRIER_DISPLAY_ID']
Xt = delivery_analysis[features]
y = delivery_analysis['ON_TIME']

# Encode categorical features
Xe = pd.get_dummies(Xt, columns=['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'SHIPMENT_NUMBER', 'CARRIER_DISPLAY_ID'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Xe, y, test_size=0.2, train_size=0.8, random_state=42)

# Train a RandomForestClassifier
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators = 100,
                           random_state = 42,
                           min_samples_split = 10,
                           max_features = "sqrt",
                           bootstrap = True)

model.fit(X_train, y_train)

# Predict on the test set and evaluate the model's performance
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict delays for new bookings
# Prepare the new bookings dataset similarly to the training set
X_new = pd.get_dummies(new_bookings, columns=['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'SHIPMENT_NUMBER', 'CARRIER_DISPLAY_ID'])

# Ensure the new data has the same structure as the training data
X_new = X_new.reindex(columns=Xe.columns, fill_value=0)

# Predict the likelihood of delay for the new bookings
new_bookings['delay_likelihood'] = model.predict(X_new)

# Display the predictions for new bookings
print("Predictions for new bookings:")
print(new_bookings[['SHIPMENT_NUMBER', 'delay_likelihood']])