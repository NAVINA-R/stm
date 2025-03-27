# Add this at the VERY TOP of ride_sharing_optimizer.py
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import seaborn as sns
from geopy.distance import great_circle
from sklearn.neighbors import NearestNeighbors

# Step 1: Data Loading & Preprocessing
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    df['Preferred_Pickup_Time'] = pd.to_datetime(df['Preferred_Pickup_Time'])
    df['Preferred_Dropoff_Time'] = pd.to_datetime(df['Preferred_Dropoff_Time'])
    
    # Extract time features
    df['Pickup_Hour'] = df['Preferred_Pickup_Time'].dt.hour
    df['Pickup_Day'] = df['Preferred_Pickup_Time'].dt.dayofweek
    df['Trip_Duration_Min'] = (df['Preferred_Dropoff_Time'] - df['Preferred_Pickup_Time']).dt.total_seconds() / 60
    
    # Clean POI data (extract numeric part)
    df['POI_Numeric'] = df['POI'].str.extract(r'(\d+)').astype(float)
    df['Route_Numeric'] = df['Preferred_Route'].str.extract(r'(\d+)').astype(float)
    return df

# Step 2: High-Demand Zone Prediction
def predict_high_demand_zones(df):
    # Analyze congestion hotspots
    hotspot_counts = df['Congestion_Hotspots'].value_counts().head(10)
    
    # Get top POIs with most rides
    top_pois = df['POI_Numeric'].value_counts().head(10).index
    
    # Time-based demand analysis
    hourly_demand = df.groupby('Pickup_Hour').size()
    daily_demand = df.groupby('Pickup_Day').size()
    
    # Spatial clustering for high-demand zones
    # We'll use POI and Distance_Matrix as spatial features
    spatial_features = df[['POI_Numeric', 'Distance_Matrix']].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(spatial_features)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    # Add clusters back to dataframe
    spatial_features['Cluster'] = clusters
    high_demand_clusters = spatial_features[spatial_features['Cluster'] != -1]
    
    return {
        'hotspot_counts': hotspot_counts,
        'top_pois': top_pois,
        'hourly_demand': hourly_demand,
        'daily_demand': daily_demand,
        'high_demand_clusters': high_demand_clusters
    }

# Step 3: Carpool Matching
def find_carpool_matches(df, n_matches=5):
    # Prepare data for matching
    df['Route_Numeric'] = df['Preferred_Route'].str.extract('(\d+)').astype(float)
    
    # Features for matching: Route, Pickup time window, Dropoff time window, POI
    features = df[['User_ID', 'Route_Numeric', 'Pickup_Hour', 'POI_Numeric', 
                   'Preferred_Pickup_Time', 'Preferred_Dropoff_Time', 'Carpool_Willingness']]
    
    # Only consider users willing to carpool
    willing_users = features[features['Carpool_Willingness'] == 'Yes']
    
    if len(willing_users) < 2:
        return pd.DataFrame()  # Not enough willing users
    
    # Convert datetime to minutes since midnight for comparison
    willing_users = willing_users.copy()
    willing_users.loc[:, 'Pickup_Min'] = willing_users['Preferred_Pickup_Time'].dt.hour * 60 + willing_users['Preferred_Pickup_Time'].dt.minute
    willing_users.loc[:, 'Dropoff_Min'] = willing_users['Preferred_Dropoff_Time'].dt.hour * 60 + willing_users['Preferred_Dropoff_Time'].dt.minute
    # Features for matching algorithm
    X_match = willing_users[['Route_Numeric', 'Pickup_Min', 'POI_Numeric']].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_match)
    
    # Use Nearest Neighbors to find similar routes/times
    nbrs = NearestNeighbors(n_neighbors=min(n_matches+1, len(willing_users)), algorithm='ball_tree').fit(X_scaled)
    
    # Find matches for each user (excluding themselves)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # Prepare matches dataframe
    matches = []
    for i in range(len(willing_users)):
        user_id = willing_users.iloc[i]['User_ID']
        for j in range(1, len(indices[i])):  # Skip first match (itself)
            match_id = willing_users.iloc[indices[i][j]]['User_ID']
            distance = distances[i][j]
            matches.append({
                'User_ID': user_id,
                'Match_ID': match_id,
                'Similarity_Score': 1/(1 + distance),  # Convert distance to similarity
                'Route': willing_users.iloc[i]['Route_Numeric'],
                'Pickup_Time': willing_users.iloc[i]['Preferred_Pickup_Time'].strftime('%H:%M'),
                'Match_Pickup_Time': willing_users.iloc[indices[i][j]]['Preferred_Pickup_Time'].strftime('%H:%M')
            })
    
    matches_df = pd.DataFrame(matches)
    
    # Sort by similarity score
    matches_df = matches_df.sort_values(by='Similarity_Score', ascending=False)
    
    return matches_df.head(50)  # Return top 50 matches

# Step 4: Visualization
def visualize_results(demand_results, carpool_matches, save_paths=None):
    sns.set(style="whitegrid")
    
    # Create figures directory if it doesn't exist
    if save_paths:
        import os
        os.makedirs(os.path.dirname(save_paths['hotspot_plot']), exist_ok=True)
    
    # 1. Plot top congestion hotspots
    plt.figure(figsize=(12, 6))
    demand_results['hotspot_counts'].plot(kind='bar')
    plt.title('Top 10 Congestion Hotspots')
    plt.xlabel('Hotspot')
    plt.ylabel('Number of Rides')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_paths:
        plt.savefig(save_paths['hotspot_plot'])
        plt.close()
    else:
        plt.show()
    
    # 2. Plot hourly demand
    plt.figure(figsize=(12, 6))
    demand_results['hourly_demand'].plot(kind='line', marker='o')
    plt.title('Ride Demand by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Rides')
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    
    if save_paths:
        plt.savefig(save_paths['hourly_plot'])
        plt.close()
    else:
        plt.show()
    
    # 3. Plot carpool matches (if any)
    if not carpool_matches.empty:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=carpool_matches, x='Route', y='Similarity_Score', hue='Pickup_Time')
        plt.title('Potential Carpool Matches by Route and Similarity Score')
        plt.xlabel('Route Number')
        plt.ylabel('Similarity Score')
        plt.tight_layout()
        plt.show()

# Main Execution
def main():
    # Load and preprocess data
    file_path = 'ride_sharing_dataset.csv'
    df = load_and_preprocess_data(file_path)
    
    # Predict high-demand zones
    demand_results = predict_high_demand_zones(df)
    
    # Find carpool matches
    carpool_matches = find_carpool_matches(df)
    
    # Visualize results
    visualize_results(demand_results, carpool_matches)
    
    # Print top carpool matches
    if not carpool_matches.empty:
        print("\nTop Carpool Matches:")
        print(carpool_matches[['User_ID', 'Match_ID', 'Similarity_Score', 'Route', 'Pickup_Time', 'Match_Pickup_Time']].head(10))
    else:
        print("\nNo suitable carpool matches found.")
    
    # Print high-demand clusters
    print("\nHigh-Demand Clusters Summary:")
    print(demand_results['high_demand_clusters'].groupby('Cluster').mean())

if __name__ == "__main__":
    main()