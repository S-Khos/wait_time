from lane import Lane
from collections import deque
from scipy.optimize import linear_sum_assignment
from lane_clustering import DBSCAN
from moving_average import MovingAverage
import numpy as np

class LaneExtractor():

    def __init__(self) -> None:
        self.n_lanes = 0
        self.lanes = None
        self.scanner = None
        self.trajectories = deque(maxlen=80)
        self.frame_size = None
        self.extraction_count = 0
        self.lane_clusters = None
        self.lane_assignments = {}
        self.total_mean_outflow_timer = MovingAverage(window_size=15)
        self.total_mean_outflow_time = 0
        self.lane_colours = [
            (30, 70, 200),    # Blue
            (160, 160, 160),  # Light gray
            (20, 80, 20),     # Dark green
            (220, 180, 40),   # Yellow
            (100, 20, 150),   # Deep purple
            (220, 100, 150),  # Pink
            (40, 150, 140),   # Teal
            (130, 70, 20),    # Brown
            (40, 50, 100),    # Dark slate blue
            (150, 30, 70)     # Burgundy
        ]

    def initialize_lanes(self, frame_size):
        self.frame_size = frame_size
        self.scanner = DBSCAN(frame_size=self.frame_size)
        self.n_lanes, self.lane_clusters = self.scanner.fit(self.trajectories)
        # self.lanes = {id: Lane(id, self.lane_clusters[id]) for id in range(1, self.n_lanes + 1)}
        self.lanes = {id: Lane(id, lane_cluster) for id, lane_cluster in self.lane_clusters.items()}
        with open("trajectories.txt", "w") as f:
            for trajectory in self.trajectories:
                f.write(str(trajectory) + ",\n")
        self.trajectories.clear()

    # def assign_vehicles(self, detections, elapsed_time):
    #     if self.scanner is None:
    #         return detections

    #     for lane in self.lanes.values():
    #         lane.vehicles.clear()
    #         lane.active_vehicle_count = 0

    #     valid_detections = [
    #         vehicle
    #         for vehicle in detections
    #         if vehicle.yFlow <= 0  # and vehicle is not lost 
    #         and not vehicle.processed
    #         and vehicle.time_waited >= 5
    #     ]

    #     if not valid_detections:
    #         return detections
    
    #     centroids = [vehicle.centroid for vehicle in valid_detections]
    #     labels = self.scanner.predict(centroids)
    #     for label, vehicle in zip(labels, valid_detections):
    #         vehicle.lane_id = self.lane_assignments.get(int(label), int(label))
    #         if vehicle.lane_id >= 1:
    #             self.lanes[vehicle.lane_id].add_vehicle(vehicle)
    #             self.lanes[vehicle.lane_id].increment_vehicle_count(vehicle)
    #         else:
    #             vehicle.processing_timer_start = None
    #             vehicle.processing_timer = 0

    #     for lane in self.lanes.values():
    #         lane.sort_vehicles()
    #         lane.start_processing_timer(elapsed_time)
    #         lane.update_total_wait_time()
    #         lane.update_total_processing_time()

    #     return detections
    def assign_vehicles(self, detections, elapsed_time):
        if self.scanner is None:
            return detections

        if not detections:
            return detections
            
        # Clear vehicles and counts only once - moved outside the lane loop
        vehicles_by_lane = {}
        
        valid_detections = [
            vehicle
            for vehicle in detections
            if vehicle.yFlow <= 0 and not vehicle.processed and vehicle.time_waited >= 5
        ]

        if not valid_detections:
            return detections
        
        # Batch process all centroids at once
        centroids = np.array([vehicle.centroid for vehicle in valid_detections])
        labels = self.scanner.predict(centroids)
        
        # Group vehicles by lane_id for batch processing
        for label, vehicle in zip(labels, valid_detections):
            # get lane_id correction
            lane_id = self.lane_assignments.get(int(label), int(label))
            vehicle.lane_id = lane_id
            
            if lane_id >= 1:
                if lane_id not in vehicles_by_lane:
                    vehicles_by_lane[lane_id] = []
                vehicles_by_lane[lane_id].append(vehicle)
            else:
                vehicle.processing_timer_start = None
                vehicle.processing_timer = 0

        # Process lanes in batch
        for lane_id, lane in self.lanes.items():
            # Clear previous state
            lane.vehicles.clear()
            lane.active_vehicle_count = 0
            
            # Add new vehicles if any
            if lane_id in vehicles_by_lane:
                lane_vehicles = vehicles_by_lane[lane_id]
                # Add all vehicles at once
                lane.vehicles.extend(lane_vehicles)
                lane.active_vehicle_count += len(lane_vehicles)
                
            # Process lane
            lane.sort_vehicles()
            lane.start_processing_timer(elapsed_time)
            lane.update_total_wait_time()
            lane.update_total_processing_time()

        return detections
    def update_total_mean_outflow_time(self):
        lane_mean_outflow_times = [lane.avg_outflow_time for lane in self.lanes.values() if lane.avg_outflow_time > 0]
        self.total_mean_outflow_timer.add(np.mean(lane_mean_outflow_times))
        self.total_mean_outflow_time = self.total_mean_outflow_timer.average()

    def prune_lanes(self):
        for lane in self.lanes.values():
            if lane.temp_outflow_count == lane.outflow_count:
                self.lanes.pop(lane.id)
                self.lane_assignments = {key: value for key, value in self.lane_assignments.items() if value != lane.id}
                self.n_lanes -= 1

    def update_lanes(self, frame_size):
            self.extraction_count += 1
            if not self.trajectories:
                return
            if self.scanner is None:
                self.initialize_lanes(frame_size)
            else:

                self.n_lanes, self.lane_clusters = self.scanner.fit(self.trajectories)
                new_lane_clusters_mean = {id: np.mean(cluster, axis=0) for id, cluster in self.lane_clusters.items()}
                # Build list of lanes and cluster ids for indexing
                lanes_list = list(self.lanes.values())
                cluster_ids = list(new_lane_clusters_mean.keys())

                # Create cost matrix: each entry is the Euclidean distance between the lane mean and the new cluster mean
                cost_matrix = np.zeros((len(lanes_list), len(cluster_ids)))
                for i, lane in enumerate(lanes_list):
                    lane_mean = lane.mean
                    for j, cluster_id in enumerate(cluster_ids):
                        cluster_mean = new_lane_clusters_mean[cluster_id]
                        cost_matrix[i, j] = np.linalg.norm(lane_mean - cluster_mean)

                # Solve for the global optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                existing_lane_ids = []
                # Update lanes based on optimal assignment
                for i, j in zip(row_ind, col_ind):
                    lane = lanes_list[i]
                    new_lane_id = cluster_ids[j]
                    self.lane_assignments[new_lane_id] = lane.id
                    # self.lanes[new_lane_id] = self.lanes.pop(lane.id)
                    existing_lane_ids.append(lane.id)
                    lane.update_lane(self.lane_clusters[new_lane_id])
                
                # existing_lane_ids.sort()
                sorted_id = list(self.lanes.keys())

                # Get unmatched clusters
                matched_cluster_ids = [cluster_ids[j] for j in col_ind]
                remaining_new_ids = [id for id in new_lane_clusters_mean.keys() if id not in matched_cluster_ids]
                for new_id in remaining_new_ids:
                    self.lanes[sorted_id[-1] + 1] = Lane(sorted_id[-1] + 1, self.lane_clusters[new_id])
                    self.lane_assignments[new_id] = sorted_id[-1] + 1

                self.trajectories.clear()

    def add_trajectories(self, trajectory):
        self.trajectories.append(trajectory)