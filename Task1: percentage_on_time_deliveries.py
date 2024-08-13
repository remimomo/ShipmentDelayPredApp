import pandas as pd
from datetime import datetime, timedelta
# !pip install numpy==1.16.0
# import json
# import numpy as np

# Load the datasets
# shipment_bookings = pd.read_csv('data/Shipment_bookings.csv')
# gps_data = pd.read_csv('data/GPS_data.csv')
shipment_bookings = pd.read_csv('/Users/remimomo/Documents/shipment_delay_prediction_v1/data/Shipment_bookings.csv')
gps_data = pd.read_csv('/Users/remimomo/Documents/shipment_delay_prediction_v1/data/GPS_data.csv')

# Convert necessary columns to datetime format
shipment_bookings['LAST_DELIVERY_SCHEDULE_EARLIEST'] = pd.to_datetime(shipment_bookings['LAST_DELIVERY_SCHEDULE_EARLIEST'])
shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'])
gps_data['RECORD_TIMESTAMP'] = pd.to_datetime(gps_data['RECORD_TIMESTAMP'])

# Filter shipments within the specified time period
start_date = pd.Timestamp("2023-10-1", tz="UTC").replace(microsecond=0)
end_date = pd.Timestamp("2023-12-31", tz="UTC").replace(microsecond=0)
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
# Shipments are on-time if the actual delivery time is within 30 minutes of the scheduled latest delivery time
delivery_analysis['ON_TIME'] = delivery_analysis['ACTUAL_DELIVERY_TIME'] <= (delivery_analysis['LAST_DELIVERY_SCHEDULE_LATEST'] + timedelta(minutes=30))

# Calculate the percentage of on-time deliveries
on_time_percentage = delivery_analysis['ON_TIME'].mean() * 100
print(f"Percentage of on-time deliveries: {on_time_percentage:.2f}%")


