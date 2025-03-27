import pickle
import numpy as np

try:
    with open('traffic_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    test_input = np.array([[50, 10, 2]])  # traffic_volume, queue_length, pedestrians
    print("Model features expected:", getattr(model, 'feature_names_in_', 'Unknown'))
    print("Prediction:", model.predict(test_input))
except Exception as e:
    print("Error testing model:", str(e))