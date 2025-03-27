from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import logging
import os


app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and sample data
try:
    with open('traffic_model.pkl', 'rb') as f:
        model = pickle.load(f)
        logger.info("Production model loaded successfully")
        
    with open('sample_data.pkl', 'rb') as f:
        sample_data = pickle.load(f)
        logger.info("Sample data loaded successfully")
        
except Exception as e:
    logger.error(f"Loading failed: {str(e)}")
    model = None
    sample_data = None

def prepare_features(input_data):
    """Prepare all features from form inputs"""
    now = datetime.now()
    features = {
        'Vehicle_Count': float(input_data.get('vehicle_count', 0)),
        'Weighted_Vehicle_Count': float(input_data.get('vehicle_count', 0)) * float(input_data.get('vehicle_type', 1.0)),
        'Speed_kmh': float(input_data.get('speed', 40)),
        'Density': float(input_data.get('density', 0.8)),
        'Number_of_Lanes': float(input_data.get('lanes', 2)),
        'Pedestrian_Crossing': float(input_data.get('pedestrian', 0)),
        'Bicycle_Lane': float(input_data.get('bike_lane', 0)),
        'Hour': now.hour,
        'DayOfWeek': now.weekday(),
        'Historical_Vehicle_Count': float(input_data.get('historical', 0)),
        'Current_Signal_Phase': 0.0  # Default value
    }
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Prediction request: {data}")
        
        # Prepare features
        features = prepare_features(data)
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([features], columns=model.feature_names_in_)
        
        # Make prediction
        optimal_duration = float(model.predict(feature_df)[0])
        
        # Clamp between reasonable values
        optimal_duration = max(min(optimal_duration, 120), 10)
        
        return jsonify({
            "status": "success",
            "optimal_duration": optimal_duration,
            "features_used": features,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/control_signal', methods=['POST'])
def control_signal():
    try:
        data = request.get_json()
        
        # Validate input
        if 'duration' not in data:
            return jsonify({"error": "Missing duration parameter"}), 400
            
        duration = float(data['duration'])
        
        return jsonify({
            "status": "success",
            "message": f"Signal controlled for {duration} seconds",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sample_data', methods=['GET'])
def get_sample_data():
    if sample_data is None:
        return jsonify({"error": "Sample data not loaded"}), 404
        
    # Convert numpy types to native Python types
    sample = sample_data.iloc[0].to_dict()
    for k, v in sample.items():
        if isinstance(v, (np.integer, np.floating)):
            sample[k] = float(v)
            
    return jsonify({
        "status": "success",
        "sample_data": sample
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)