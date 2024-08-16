from flask import Flask, render_template, jsonify
import plotly.express as px
import pandas as pd
import json

app = Flask(__name__)

# Function to simulate getting real-time data
def get_real_time_data():
    # In a real implementation, this would pull data from the prediction model, traffic, and weather APIs
    data = {
        "prediction": "On Time",
        "traffic": "Moderate",
        "weather": "Clear",
        "temperature": 22
    }
    return data

# Route for the dashboard
@app.route('/dashboard')
def dashboard():
    data = get_real_time_data()
    
    # Example: Create a bar chart for traffic and weather conditions
    df = pd.DataFrame({
        'Condition': ['Traffic', 'Weather'],
        'Status': [data['traffic'], data['weather']]
    })
    fig = px.bar(df, x='Condition', y='Status')
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('dashboard.html', graph_json=graph_json, data=data)

if __name__ == "__main__":
    app.run(debug=True)
