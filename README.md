## Directory Structure

shipment_delay_prediction/
├── data/
│   ├── GPS_data.csv
│   ├── Shipment_bookings.csv
│   ├── New_bookings.csv
├── src/
│   ├── app.py
│   ├── model.py
│   ├── data_processing.py
│   ├── api_integration.py
│   ├── config.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   ├── static/
│   │   ├── styles.css
│   ├── tests/
│   │   ├── test_model.py
│   │   └── test_data_processing.py
├── requirements.txt
├── Procfile
├── README.md
└── .env

# SHIPMENT DELAY PREDICTOR

This Python web application uses a pre-trained model to predict the likelihood of delays for shipments. The application integrates real-time weather data to enhance prediction accuracy.

## Features
- Predict shipment delays based on input features and real-time weather conditions.
- Simple and intuitive web interface.
- Unit tests to ensure app functionality.

## Prerequisites
- Python 3.7 or higher
- Flask
- Pandas
- scikit-learn
- Requests (for API calls)

## Setup and Running the Application

##1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/shipment_delay_predictor.git
   cd shipment_delay_predictor
2.	## Install Required Packages:

pip install -r requirements.txt
3.	## Set Up Your API Key:
o	Sign up for an API key from OpenWeatherMap.
o	Replace "your_openweathermap_api_key" in app.py with your actual API key.
4.	## Run the Flask Application:
Copy code
python app.py
5.	## Visit the Web Application: Open your web browser and visit http://127.0.0.1:5000/.
6 ## Running Tests
To run the unit tests, execute the following command:
bash
Copy code
python -m unittest discover -s tests

7 ##Run the Flask Application Locally. Test the API locally using curl:
Copy code
curl -X POST http://127.0.0.1:5000/predict_delay \
    -H "Content-Type: application/json" \
    -d '{
    "VEHICLE_SIZE": "Large",
    "VEHICLE_BUILD_UP": "Refrigerated",
    "FIRST_COLLECTION_LATITUDE": 51.5074,
    "FIRST_COLLECTION_LONGITUDE": -0.1278,
    "FIRST_COLLECTION_SCHEDULE_EARLIEST": "2023-10-01T08:00:00",
    "FIRST_COLLECTION_SCHEDULE_LATEST": "2023-10-01T09:00:00",
    "LAST_DELIVERY_SCHEDULE_EARLIEST": "2023-10-01T14:00:00",
    "LAST_DELIVERY_SCHEDULE_LATEST": "2023-10-01T15:00:00"
}'

8 ## Deployment to Heroku (Optional)
Follow the same steps as previously mentioned for deploying the Flask app to Heroku:
1.	Initialise Git:
Copy code
git init
git add .
git commit -m "Initial commit with external API integration"
2.	Deploy to Heroku:
Copy code
heroku create your-app-name

## Setting up  and running a CI/CD pipeline
Step 1: Set Up a GitHub Repository
Create a GitHub repository and push your local project to GitHub:

git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin master

Step 2: Create a .github/workflows Directory
In your project’s root directory, create a directory for your GitHub Actions workflow:
mkdir -p .github/workflows

Step 3: Create a GitHub Actions Workflow
Inside the .github/workflows/ directory, create a YAML file for your CI/CD pipeline. 
Step 4: Add Heroku API Key to GitHub Secrets
To securely store your Heroku API key:
1.	Go to your GitHub repository.
2.	Click on Settings > Secrets and variables > Actions.
3.	Click New repository secret.
4.	Name the secret HEROKU_API_KEY and paste your Heroku API key as the value.
Step 5: Configure Tests and Linting (Optional)
If you haven’t already, you might want to add basic tests and linting to your project:
1.	Create a tests/ directory with test files if you don't have tests set up.
2.	Add flake8 configuration to your setup.cfg or tox.ini if you want to customize linting.
Step 6: Push Changes to GitHub
Push your changes to the master branch to trigger the CI/CD pipeline:
git add .
git commit -m "Set up CI/CD pipeline with GitHub Actions"
git push origin master

Step 7: Monitor the Pipeline
1.	Go to your GitHub repository.
2.	Click on the Actions tab.
3.	You should see the CI/CD Pipeline running. If everything is set up correctly, the pipeline will:
o	Run linting and tests.
o	Deploy your application to Heroku if the tests pass.
Step 8: Verify Deployment on Heroku
After the deployment step completes, you can visit your Heroku app’s URL (e.g., https://your-heroku-app-name.herokuapp.com/) to verify that your changes have been deployed.
