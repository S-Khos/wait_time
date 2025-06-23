import numpy as np
import os
from fixed_buffer import FixedBuffer
from collections import deque
from util import get_color_name

class Vehicle():
    class_names = []
    def __init__(self, bbox, track_id, class_id, elapsed_time, frame_size, max_traj=45):
        # Handle different bbox formats - ONVIF gives [x, y, w, h] format
        if hasattr(bbox, 'int'):
            x, y, width, height = bbox.int().tolist()
        else: 
            x, y, width, height = bbox.astype(int)
        
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        self.x1 = int(x)
        self.y1 = int(y)
        self.x2 = int(x + width)
        self.y2 = int(y + height)
        self.centroid = (int((self.x1 + self.x2) // 2), int((self.y1 + self.y2) // 2))
        self.id = track_id
        self.lane_id = -1
        self.time_entered = elapsed_time
        self.time_waited = 0
        self.processing_timer_start = None
        self.processing_timer = 0
        self.stopped = False
        self.stop_timer = 0
        self.lost = False
        self.lost_timer = 0
        self.yFlow = 0
        self.xFlow = 0
        self.positive_yFlow = 0
        self.negative_yFlow = 0
        self.positive_xFlow = 0
        self.negative_xFlow = 0
        self.processed = False
        self.outflow_line = None
        self.colour = "NA"
        self.trajectory = deque(maxlen=max_traj)
        self.frame_size = frame_size
        self.traj_dist = int(24 * ((self.frame_size[0] * self.frame_size[1]) // (2592 * 1944)))
        self.stop_dist_threshold = self.traj_dist
        # self.trajectory = FixedBuffer(max_len=max_traj)
        self.colour_hist = None
        self.bbox_ratio = 0
        self.max_traj = max_traj
        self.trajectory.append(self.centroid)
        self.dist_prev_pos = 0
        if not Vehicle.class_names:
            classes_path = f"./configs/coco.names"
            if not os.path.exists(classes_path):
                logger.error(
                    f"Unable to load class names, {classes_path}, in given directory"
                )
                raise FileNotFoundError(
                    f"Unable to load class names, {classes_path}, in given directory"
                )

            with open(classes_path, "r") as f:
                Vehicle.class_names = f.read().strip().split("\n")
        self.class_id = class_id

    def update_pos(self, bbox) -> None:
        # Handle different bbox formats - ONVIF gives [x, y, w, h] format
        if hasattr(bbox, 'int'):
            x, y, width, height = bbox.int().tolist()
        else:
            x, y, width, height = bbox.astype(int)
            
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        self.x1 = int(x)
        self.y1 = int(y)
        self.x2 = int(x + width)
        self.y2 = int(y + height)
        self.centroid = (int((self.x1 + self.x2) // 2), int((self.y1 + self.y2) // 2))
        self.calculate_bbox_ratio()

    def update_time_waited(self, elapsed_time) -> None:
        self.time_waited = elapsed_time - self.time_entered

    def update_flow(self) -> None:
        if len(self.trajectory) < 2:
            return
        # get y flow
        y_coords = np.array(self.trajectory)[-20:, 1]
        y_diff = np.diff(y_coords)
        self.negative_yFlow += np.sum(y_diff < 0)
        self.positive_yFlow += np.sum(y_diff > 0)
        self.yFlow = np.sign(self.positive_yFlow - self.negative_yFlow) * -1

        # get x flow
        x_coords = np.array(self.trajectory)[-20:, 0]
        x_diff = np.diff(x_coords)
        self.negative_xFlow += np.sum(x_diff < 0)
        self.positive_xFlow += np.sum(x_diff > 0)
        self.xFlow = np.sign(self.positive_xFlow - self.negative_xFlow) * -1

    def update_processing_timer(self, elapsed_time) -> None:
        if not self.lost:
            if self.processing_timer_start is None:
                self.processing_timer_start = elapsed_time
            self.processing_timer = elapsed_time - self.processing_timer_start

    def get_lost_timer(self, elapsed_time) -> float:
        if self.lost:
            if self.lost_timer == 0:
                self.lost_timer = elapsed_time
                return 0
            return elapsed_time - self.lost_timer
        else:
            return 0
        
    def get_stop_timer(self, elapsed_time) -> float:
        if self.dist_prev_pos <= self.stop_dist_threshold and not self.lost:
            if self.stop_timer == 0:
                self.stop_timer = elapsed_time
                return 0
            return elapsed_time - self.stop_timer
        else:
            self.stop_timer = 0
            return 0

    def passed_outflow_line(self) -> bool:
        if self.outflow_line is not None:
            if ((self.centroid[1] > self.outflow_line[1]) 
            and ((self.x1 >= self.outflow_line[0] and self.x1 <= self.outflow_line[2])
                or (self.x2 >= self.outflow_line[0] and self.x2 <= self.outflow_line[2]))):
                return True
        return False
        
    def update_trajectory(self):
        if len(self.trajectory) == 0 or self.trajectory[-1] != self.centroid:
            if len(self.trajectory) > 0:
                last_centroid = np.array(self.trajectory[-1])
                self.dist_prev_pos = np.linalg.norm(np.array(self.centroid) - last_centroid)
                if self.dist_prev_pos > self.traj_dist:
                    self.trajectory.append(self.centroid)
            else:
                self.trajectory.append(self.centroid)

    def update_lane_id(self, lane_id) -> None:
        self.lane_id = lane_id

    def calculate_bbox_ratio(self) -> None:
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        self.bbox_ratio = width / height if height > 0 else 0

    def is_semi_truck(self) -> bool:
        return self.bbox_ratio > 1.87 and self.class_id == "truck" # and white
    
    def is_valid(self):
        pass
    
    def get_traj_std(self):
        if len(self.trajectory) < 2:
            return (0,0)
        
        return (np.std(np.array(self.trajectory)[-10:], axis=0))

    def get_colour(self, frame) -> None:
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        roi = frame[y1:y2, x1:x2]
        # Calculate the average color in the ROI
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color = tuple(avg_color.astype(int))
        # Convert the average color to a color name
        self.colour = get_color_name(avg_color)


