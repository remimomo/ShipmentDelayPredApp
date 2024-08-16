###########1. Load the Data ############
import pandas as pd
import numpy as np

# Load the dataset
# Assuming the dataset is named 'shipment_data.csv' for this analysis
#shipment_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Shipment_bookings.csv')
shipment_data = pd.read_csv('/Data/Shipment_bookings.csv')

# Display the first few rows to understand the structure
shipment_data.head()

########### 2. Data Cleaning ############
# Check for missing values
missing_values = shipment_data.isnull().sum()
print("Missing Values:\n", missing_values)

# If there are missing values, decide how to handle them
# For example, fill with median or mode, or drop if appropriate
shipment_data = shipment_data.dropna()  # For simplicity, let's drop missing values

# Check for duplicates
duplicates = shipment_data.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# If duplicates exist, remove them
shipment_data = shipment_data.drop_duplicates()

# Convert date columns to datetime format for easier manipulation
shipment_data['FIRST_COLLECTION_SCHEDULE_EARLIEST'] = pd.to_datetime(shipment_data['FIRST_COLLECTION_SCHEDULE_EARLIEST'])
shipment_data['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(shipment_data['LAST_DELIVERY_SCHEDULE_LATEST'])

########## 3. Descriptive Statistics ##########
# Get summary statistics for numerical columns
summary_statistics = shipment_data.describe()
print("Summary Statistics:\n", summary_statistics)

# Get summary statistics for categorical columns
categorical_summary = shipment_data.describe(include=['object'])
print("Categorical Summary:\n", categorical_summary)

# Identify the target variable (e.g., delay status, if exists in the dataset)
# Assuming the target variable is 'DELAY_STATUS' with 0 = On Time, 1 = Delayed

########## 4. Correlation Analysis ##########
import pandas as pd

# Assuming the dataset includes an actual delivery time column, 'ACTUAL_DELIVERY_TIME'
# If not present, we would need to merge GPS data or equivalent to get this column.

# Example dataset with 'ACTUAL_DELIVERY_TIME' and 'LAST_DELIVERY_SCHEDULE_LATEST'
# For this example, I will create a small DataFrame to demonstrate
data = {
    'SHIPMENT_NUMBER': [1, 2, 3, 4, 5],
    'ACTUAL_DELIVERY_TIME': pd.to_datetime([
        '2023-08-01 18:30', '2023-08-01 19:30', '2023-08-01 19:45', '2023-08-01 20:15', '2023-08-01 21:30'
    ]),
    'LAST_DELIVERY_SCHEDULE_LATEST': pd.to_datetime([
        '2023-08-01 18:00', '2023-08-01 19:00', '2023-08-01 20:00', '2023-08-01 20:00', '2023-08-01 22:00'
    ])
}

shipment_data = pd.DataFrame(data)

# Calculate DELAY_STATUS: 0 for On Time, 1 for Delayed
shipment_data['DELAY_STATUS'] = (shipment_data['ACTUAL_DELIVERY_TIME'] > shipment_data['LAST_DELIVERY_SCHEDULE_LATEST']).astype(int)

# Display the DataFrame to show the DELAY_STATUS calculation
shipment_data

# Convert categorical variables to numerical for correlation analysis
# Using one-hot encoding for categorical variables
shipment_data_encoded = pd.get_dummies(shipment_data, drop_first=True)

# Calculate the correlation matrix
correlation_matrix = shipment_data_encoded.corr()

# Display the correlation matrix for the target variable 'DELAY_STATUS'
correlation_with_delay = correlation_matrix['DELAY_STATUS'].sort_values(ascending=False)
print("Correlation with Delay Status:\n", correlation_with_delay)

########## 5. Visualisation ##########
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(shipment_data, hue='DELAY_STATUS')
plt.show()

# Boxplot for each feature by 'DELAY_STATUS'
for column in shipment_data.columns:
    if shipment_data[column].dtype != 'object':
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='DELAY_STATUS', y=column, data=shipment_data)
        plt.title(f'Boxplot of {column} by Delay Status')
        plt.show()

########## 6. Feature Importance using RandomForest #########
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
import joblib

# Load the datasets
shipment_bookings = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Shipment_bookings.csv')
gps_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GPS_data.csv')
new_bookings = pd.read_csv('/content/drive/My Drive/Colab Notebooks/New_bookings.csv')

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

# Step 1: Convert relevant columns to datetime format
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

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get feature importances
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': Xe.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", importance_df)

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from RandomForest')
plt.show()