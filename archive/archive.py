import numpy as np
import cv2

# Simulated centroids of cars in two lanes (lane1, lane2)
lane1_centroids = np.array([
    [50, 350],
    [100, 300],
    [150, 250],
    [200, 180],
    [250, 120],
    [300, 60],
    [350, 30],
])

lane2_centroids = np.array([
    [50, 380],
    [100, 320],
    [150, 260],
    [200, 200],
    [250, 150],
    [300, 100],
    [350, 50],
])

# Fit polynomials for both lanes
degree = 3  # Degree of the polynomial

# Fit polynomial for lane 1
coefficients_lane1 = np.polyfit(lane1_centroids[:, 0], lane1_centroids[:, 1], degree)
polynomial_lane1 = np.poly1d(coefficients_lane1)

# Fit polynomial for lane 2
coefficients_lane2 = np.polyfit(lane2_centroids[:, 0], lane2_centroids[:, 1], degree)
polynomial_lane2 = np.poly1d(coefficients_lane2)

# Create a function to assign a vehicle to a lane
def assign_to_lane(centroid_x, centroid_y):
    # Evaluate the polynomial for both lanes
    expected_y_lane1 = polynomial_lane1(centroid_x)
    expected_y_lane2 = polynomial_lane2(centroid_x)

    # Calculate distances to each lane
    distance_to_lane1 = abs(expected_y_lane1 - centroid_y)
    distance_to_lane2 = abs(expected_y_lane2 - centroid_y)

    # Assign to the lane with the smallest distance
    if distance_to_lane1 < distance_to_lane2:
        return 'lane1', expected_y_lane1
    else:
        return 'lane2', expected_y_lane2

# Example new points (centroids from vehicles)
new_vehicle_centroids = np.array([
    [60, 360],  # This may belong to lane1
    [110, 335], # This may belong to lane2
    [200, 220], # Between lanes
    [300, 90],  # This may belong to lane2
])

# Assign each new vehicle centroid to a lane
for (cx, cy) in new_vehicle_centroids:
    assigned_lane, expected_y = assign_to_lane(cx, cy)
    print(f'Vehicle at ({cx}, {cy}) is assigned to {assigned_lane} with expected y: {expected_y:.2f}')

# Visualization section (if needed)
# Create an image to draw on
height, width = 400, 400
image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw original lanes
x_poly = np.linspace(0, width, num=400)
y_poly_lane1 = polynomial_lane1(x_poly)
y_poly_lane2 = polynomial_lane2(x_poly)

for i in range(len(x_poly) - 1):
    x1, y1 = int(x_poly[i]), int(height - np.clip(y_poly_lane1[i], 0, height))
    x2, y2 = int(x_poly[i + 1]), int(height - np.clip(y_poly_lane1[i + 1], 0, height))
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw lane1

    x1, y1 = int(x_poly[i]), int(height - np.clip(y_poly_lane2[i], 0, height))
    x2, y2 = int(x_poly[i + 1]), int(height - np.clip(y_poly_lane2[i + 1], 0, height))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw lane2

# Draw new centroids and assignments
for (cx, cy) in new_vehicle_centroids:
    lane, _ = assign_to_lane(cx, cy)
    color = (0, 255, 255) if lane == 'lane1' else (255, 255, 0)  # Different colors for different lanes
    cv2.circle(image, (int(cx), int(height - cy)), 5, color, -1)

# Show the image with lanes and vehicles
cv2.imshow('Lane Assignment', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# class LaneExtractor:
#     def __init__(self):
#         self.lanes = []
#     def assign_vehicles(self, detections):
#         trajectories = [vehicle.centroid for vehicle in detections]
#         dbscan = DBSCAN(eps=100, min_samples=2, metric=weighted_distance)
#         labels = dbscan.fit_predict(trajectories)
#         for idx, vehicle in enumerate(detections):
#             vehicle.update_lane_id(labels[idx])
#         return detections
    
    # def initialize_lanes(self, detections, feed_height):
    #     """
    #     ADD
    #     1. incorperate thershold offset based on distance (area of bounding box) of the object
    #     2. prespective ratification
    #     3. dynamic threshold based on the average size of the bounding box (width) of the mean x
    #     """
    #     queue = []
    #     lane_id = 0
    #     threshold = 125
    #     sorted_objects = sorted(
    #         [
    #             vehicle.centroid[0]
    #             for vehicle in detections
    #             if vehicle.centroid[1] >= (feed_height // 2 - 100)
    #         ]
    #     )
    #     print(sorted_objects)
    #     for idx, cur_val in enumerate(sorted_objects):
    #         queue.append(cur_val)
    #         if idx == len(sorted_objects) - 1:
    #             mean = np.mean(queue)
    #             lane = Lane(lane_id, mean)
    #             self.lanes.append(lane)
    #             lane_id += 1
    #             queue = []
    #             break
    #         if abs(sorted_objects[idx + 1] - np.mean(queue)) >= threshold:
    #             mean = np.mean(queue)
    #             lane = Lane(lane_id, mean)
    #             self.lanes.append(lane)
    #             lane_id += 1
    #             queue = []

    #     return self.lanes
