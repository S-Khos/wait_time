from moving_average import MovingAverage
import numpy as np

class Lane():
    def __init__(self, id, cluster):
        self.id = id
        self.vehicles = []
        self.avg_wait_time = 0
        self.total_wait_time = 0
        self.avg_outflow_time = 0
        self.total_outflow_time = 0
        self.cur_outflow_time = 0
        self.wait_times = MovingAverage(window_size=15)
        self.outflow_times = MovingAverage(window_size=15)
        self.temp_outflow_count = 0
        self.outflow_count = 0
        self.active_vehicle_count = 0
        self.cluster = np.array(cluster)
        self.previous_lane_count = 0
        self.mean = np.mean(self.cluster, axis=0)
        self.std = np.std(self.cluster, axis=0)

    
    def clear(self):
        self.vehicles = []
        self.avg_wait_time = 0
        self.total_wait_time = 0
        self.avg_outflow_time = 0
        self.total_outflow_time = 0
        self.cur_outflow_time = 0
        self.wait_times.clear()
        self.outflow_times.clear()
        self.temp_outflow_count = 0
        self.outflow_count = 0
        self.active_vehicle_count = 0
        self.previous_lane_count = 0

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def increment_vehicle_count(self, vehicle):
        if not vehicle.lost and vehicle.yFlow <= 0:
            self.active_vehicle_count += 1

    def sort_vehicles(self):
        self.vehicles.sort(key=lambda vehicle: vehicle.centroid[1], reverse=True)

    def start_processing_timer(self, elapsed_time):
        if not self.vehicles:
            print(f"No vehicles in lane {self.id}")
            return

        if self.vehicles[0].time_waited >= 5:
            self.vehicles[0].update_processing_timer(elapsed_time)
            self.cur_outflow_time = self.vehicles[0].processing_timer

        if len(self.vehicles) > 1:
            for vehicle in self.vehicles[1:]:
                vehicle.processing_timer_start = None
                vehicle.processing_timer = 0

    def update_total_wait_time(self):
        self.total_wait_time = self.avg_wait_time * self.active_vehicle_count

    def update_total_processing_time(self):
        self.total_outflow_time = (
            self.avg_outflow_time * self.active_vehicle_count
        )

    def update_processed_vehicles(self):
        self.outflow_count += 1

    def update_lane(self, new_cluster, new_id=None):
        # self.lane = new_id
        if self.temp_outflow_count == self.outflow_count:
            self.clear()

        self.cluster = np.array(new_cluster)
        self.lane_mean = np.mean(self.cluster, axis=0)
        self.lane_std = np.std(self.cluster, axis=0)
        self.temp_outflow_count = self.outflow_count
