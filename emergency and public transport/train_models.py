import pandas as pd
import numpy as np
import networkx as nx
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Ambulance Route Model (Graph-Based)
# --------------------------
def train_route_model(data_path: str, output_path: str):
    """Train and save a road network graph for route optimization."""
    df = pd.read_csv(data_path)
    G = nx.Graph()

    # Add edges with dynamic weights
    for _, row in df.iterrows():
        start = (row['emergency_lat'], row['emergency_lon'])
        end = (row['hospital_lat'], row['hospital_lon'])
        weight = _calculate_route_weight(row['traffic_density'], row['road_condition'])
        G.add_edge(start, end, weight=weight)

    # Save the graph
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Route model saved to {output_path}")

def _calculate_route_weight(traffic: str, condition: str) -> float:
    """Calculate edge weight based on traffic and road conditions."""
    traffic_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Gridlock': 5}
    condition_penalties = {'Dry': 1, 'Wet': 1.5, 'Accident': 3, 'Construction': 2}
    return traffic_weights[traffic] * condition_penalties.get(condition, 1)

# --------------------------
# 2. Public Transport Delay Model (XGBoost)
# --------------------------
def train_delay_model(data_path: str, output_path: str):
    """Train and save an XGBoost model for delay prediction."""
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    le = LabelEncoder()
    X = pd.DataFrame({
        'traffic': le.fit_transform(df['traffic_density']),
        'weather': le.fit_transform(df['weather'])
    })
    y = (df['public_transit_delay_mins'] > 10).astype(int)  # Binary classification

    # Train/Test Split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Delay model saved to {output_path}")

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    DATA_PATH = "data/traffic_emergency_dataset.csv"
    
    # Train and save models
    train_route_model(DATA_PATH, "models/route_model.pkl")
    train_delay_model(DATA_PATH, "models/delay_model.pkl")