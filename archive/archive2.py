import numpy as np
from sklearn.cluster import MiniBatchKMeans

class LaneExtractor:
    def __init__(self, num_lanes):
        self.lanes = []
        self.num_lanes = num_lanes
        self.kmeans = None

    def initialize_lanes(self, detections):
        vehicle_trajectories = np.array([vehicle.centroid for vehicle in detections])
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_lanes, init='k-means++', n_init=8, max_iter=10, random_state=15)
        self.kmeans.fit(vehicle_trajectories)

    def assign_vehicles(self, detections):
        if self.kmeans is None:
            raise ValueError("KMeans has not been initialized.")
        lane_ids = self.kmeans.predict(np.array([vehicle.centroid for vehicle in detections]))
        for i, vehicle in enumerate(detections):
            vehicle.lane_id = lane_ids[i]
        return detections

    def update_lanes(self, detections):
        if self.kmeans is None:
            raise ValueError("KMeans has not been initialized.")
        vehicle_trajectories = [trajectory_point for vehicle in detections for trajectory_point in vehicle.trajectory]
        self.kmeans.partial_fit(vehicle_trajectories)