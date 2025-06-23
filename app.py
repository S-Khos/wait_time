import torch
import numpy as np
import cv2
import os
import datetime
import sys
from ultralytics import YOLO, RTDETR
from vehicle import Vehicle
from lane_extractor import LaneExtractor
from util import get_time_format
import pandas as pd
import threading
import queue
import time

FRAME_WIDTH = 0
FRAME_HEIGHT = 0

class ThreadedVideoCapture:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source {source}")
        self.queue = queue.Queue(maxsize=40)
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.running = True
        self.sleep_time = 0.001
        self.thread.start()

    # In _read_frames method:
    def _read_frames(self):
        while self.running:
            if not self.queue.full():
                ret, frame = self.capture.read()
                if not ret:
                    self.running = False
                    break
                self.queue.put(frame)
                self.sleep_time = 0.001
            else:
                time.sleep(self.sleep_time)
                # Increase sleep time when queue is full (up to 50ms)
                self.sleep_time = min(self.sleep_time * 1.5, 0.05)
            

    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def release(self):
        self.running = False
        self.thread.join()
        self.capture.release()

def initialize_feed_capture(feed_source):
    return ThreadedVideoCapture(f"./feed/{feed_source}.mp4")

def create_df():
    columns = [
        "poe",
        "date",
        "time",
        "lane_id",
        "vehicle_num",
        "time_waited",
        "processing_time",
        "colour",
        "class_id",
    ]
    df = pd.DataFrame(columns=columns)
    return df

def process_detections(results, vehicles, elapsed_time, lane_extraction, df):
    global FRAME_WIDTH, FRAME_HEIGHT
    
    # Optimize: Extract all tensors at once to reduce GPU-CPU transfers
    boxes_tensor = results[0].boxes
    bboxes = boxes_tensor.xywh.cpu().numpy()
    cur_track_ids = boxes_tensor.id.int().cpu().numpy()
    cur_class_ids = boxes_tensor.cls.int().cpu().numpy()
    
    # Create sets for faster lookups
    cur_track_ids_set = set(cur_track_ids)
    vehicle_ids_set = set(vehicle.id for vehicle in vehicles)
    
    vehicles_to_remove = []
    vehicles_to_process = []
    
    for idx, vehicle in enumerate(vehicles):
        if vehicle.id not in cur_track_ids_set:
            if not vehicle.lost:
                vehicle.lost = True
                vehicle.get_lost_timer(elapsed_time)
                vehicle.update_time_waited(elapsed_time)
            else:
                if vehicle.get_lost_timer(elapsed_time) >= 5:
                    # process vehicles within lane that are out of view
                    if ((vehicle.lane_id != -1 and not vehicle.processed 
                         and len(vehicle.trajectory) >= 5) 
                         and vehicle.processing_timer >= 15
                         and vehicle.get_traj_std()[1] >= 15):
                        vehicles_to_process.append(vehicle)
                    # collect trajectories of vehicles that do not belong to any lane and are out of view
                    elif (vehicle.lane_id == -1 and 
                        vehicle.yFlow < 0 and 
                        len(vehicle.trajectory) >= 5 and 
                        vehicle.time_waited >= 15 and
                        vehicle.centroid[1] >= 0.35 * FRAME_HEIGHT 
                        and not vehicle.processed
                        and vehicle.get_traj_std()[1] >= 15):
                        lane_extraction.add_trajectories(vehicle.trajectory)
                        vehicle.processed = True
                    else:
                        vehicles_to_remove.append(idx)
                        continue
                vehicle.update_time_waited(elapsed_time)

    # Batch process vehicles that need to be added to the dataframe
    if vehicles_to_process:
        df = process_vehicles_batch(vehicles_to_process, elapsed_time, lane_extraction, df)

    # Remove vehicles in reverse order to maintain correct indices
    for idx in sorted(vehicles_to_remove, reverse=True):
        vehicles.pop(idx)

    # Create a dictionary for faster lookups
    vehicle_dict = {vehicle.id: vehicle for vehicle in vehicles}

    # Batch create new vehicles
    new_vehicles = []
    for bbox, track_id, class_id in zip(bboxes, cur_track_ids, cur_class_ids):
        if track_id in vehicle_dict:
            vehicle = vehicle_dict[track_id]
            if vehicle.lost:
                vehicle.lost = False
                vehicle.lost_timer = 0
            vehicle.update_pos(bbox)
            vehicle.update_time_waited(elapsed_time)
            vehicle.update_trajectory()
            vehicle.update_flow()
        elif track_id not in vehicle_ids_set:
            new_vehicles.append(Vehicle(bbox, track_id, class_id, elapsed_time, (FRAME_WIDTH, FRAME_HEIGHT)))
    
    # Add all new vehicles at once
    vehicles.extend(new_vehicles)

    return vehicles, lane_extraction, df

def process_vehicles_batch(vehicles_to_process, elapsed_time, lane_extraction, df):
    """Batch process vehicles that need to be added to the dataframe"""
    batch_data = []
    
    for vehicle in vehicles_to_process:
        elapsed_time_min, elapsed_time_s = get_time_format(elapsed_time)
        et_text = f"{int(elapsed_time_min)}:{int(elapsed_time_s):02d}"
        
        batch_data.append({
            "poe": "Ambassador Bridge",
            "date": datetime.datetime.now().date(),
            "time": et_text,
            "lane_id": vehicle.lane_id,
            "vehicle_num": vehicle.id,
            "time_waited": int(vehicle.time_waited),
            "processing_time": int(vehicle.processing_timer),
            "colour": vehicle.colour,
            "class_id": vehicle.class_id,
        })
        
        lane = lane_extraction.lanes[vehicle.lane_id]
        lane.outflow_times.add(vehicle.processing_timer)
        lane.wait_times.add(vehicle.time_waited)
        lane.avg_outflow_time = lane.outflow_times.average()
        lane.avg_wait_time = lane.wait_times.average()
        lane_extraction.update_total_mean_outflow_time()
        lane.update_processed_vehicles()
        lane_extraction.add_trajectories(vehicle.trajectory)
        vehicle.processing_timer = 0
        vehicle.processed = True
    
    # Batch concat new data
    if batch_data:
        new_df = pd.DataFrame(batch_data)
        return pd.concat([df, new_df], ignore_index=True)
    return df

def batch_update_vehicles(vehicles, elapsed_time):

    for vehicle in vehicles:
        vehicle.update_time_waited(elapsed_time)
        if vehicle.processing_timer > 0:
            vehicle.update_processing_timer(elapsed_time)
    return vehicles

def draw(frame, vehicles, lane_extraction, elapsed_time):
    global FRAME_WIDTH, FRAME_HEIGHT
    vehicle_telemetry_font = 0.42
    gui_font = 0.6
    for vehicle in vehicles:
        if vehicle.yFlow < 1: # not vehicle.processed:
            colour = (
                (0, 255, 0)
                if (vehicle.processing_timer > 0)
                else (
                    lane_extraction.lane_colours[vehicle.lane_id]
                    if vehicle.lane_id != -1
                    else (0, 0, 255)
                )
            )
            colour = (0, 165, 255) if vehicle.lost else colour
            cv2.putText(
                frame,
                f"CLS {vehicle.class_id.upper()}",
                (vehicle.x1, vehicle.y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"ID {vehicle.id}",
                (vehicle.x1, vehicle.y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"LN {vehicle.lane_id}",
                (vehicle.x1, vehicle.y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            # cv2.putText(
            #     frame,
            #     f"Y HDG {vehicle.yFlow}",
            #     (vehicle.x1, vehicle.y1 + 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     vehicle_telemetry_font,
            #     colour,
            #     2,
            # )
            # cv2.putText(
            #     frame,
            #     f"X HDG {vehicle.xFlow}",
            #     (vehicle.x1, vehicle.y1 + 45),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     vehicle_telemetry_font,
            #     colour,
            #     2,
            # )
            wait_time_min, wait_time_s = get_time_format(vehicle.time_waited)
            cv2.putText(
                frame,
                f"WT {int(wait_time_min)}:{int(wait_time_s):02d}",
                (vehicle.x1, vehicle.y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            # cv2.putText(
            #     frame,
            #     f"Y STD {vehicle.get_traj_std()[1]}",
            #     (vehicle.x1, vehicle.y1 + 75),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     vehicle_telemetry_font,
            #     colour,
            #     2,
            # )
            # if vehicle.processing_timer > 0:
            stop_time_min, stop_time_s = get_time_format(vehicle.get_stop_timer(elapsed_time))
            cv2.putText(
                frame,
                f"STP {int(stop_time_min)}:{int(stop_time_s):02d}",
                (vehicle.x1, vehicle.y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            ) if int(vehicle.get_stop_timer(elapsed_time)) > 0 else None
            lost_time_min, lost_time_s = get_time_format(vehicle.get_lost_timer(elapsed_time))
            cv2.putText(
                frame,
                f"LST {int(lost_time_min)}:{int(lost_time_s):02d}",
                (vehicle.x1, vehicle.y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            ) if int(vehicle.get_lost_timer(elapsed_time)) > 0 else None
            # cv2.putText(frame, f"RTO {vehicle.bbox_ratio:.2f}", (vehicle.x1, vehicle.y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, vehicle_telemetry_font, colour, 2)
            if vehicle.processing_timer > 0:
                processing_time_min, processing_time_s = get_time_format(
                    vehicle.processing_timer
                )
                cv2.putText(
                    frame,
                    f"OT {int(processing_time_min)}:{int(processing_time_s):02d}",
                    (vehicle.x2 - 20, vehicle.y1 - 15),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    colour,
                    2,
                )

            # cv2.circle(frame, vehicle.centroid, 3, colour, -1)
            if lane_extraction.lanes:
                for id, lane in lane_extraction.lanes.items():
                    color = lane_extraction.lane_colours[lane.id]
                    for pt in lane.cluster:
                        cv2.circle(frame, tuple(pt), radius=1, color=color, thickness=-1)

    elapsed_time_min, elapsed_time_s = get_time_format(elapsed_time)
    et_text = f"ELAPSED TIME {int(elapsed_time_min)}:{int(elapsed_time_s):02d}"
    cv2.putText(
        frame, et_text, (20, 30), cv2.FONT_HERSHEY_DUPLEX, gui_font, (255, 0, 0), 1
    )
    cv2.putText(
        frame,
        f"TOTAL OUTFLOW COUNT {len(lane_extraction.trajectories)}",
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        gui_font,
        (255, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        f"CLST COUNT {lane_extraction.extraction_count}",
        (20, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        gui_font,
        (255, 0, 0),
        1,
    )
    total_mean_outflow_time_m, total_mean_outflow_time_s = get_time_format(
        lane_extraction.total_mean_outflow_time
    )
    cv2.putText(
        frame,
        f"TOTAL MEAN OUTFLOW TIME {int(total_mean_outflow_time_m)}:{int(total_mean_outflow_time_s):02d}",
        (20, 90),
        cv2.FONT_HERSHEY_DUPLEX,
        gui_font,
        (255, 0, 0),
        1,
    )

    base_y = 110
    text_spacing = 25
    num_text_entries = 4
    if lane_extraction.lanes:
        for lane in lane_extraction.lanes.values():
            lane_base_y = base_y + (lane.id * text_spacing * num_text_entries)
            avg_processing_time_min, avg_processing_time_s = get_time_format(
                lane.avg_outflow_time
            )
            text = f"[{lane.id}] AVG OUTFLOW TIME {int(avg_processing_time_min)}:{int(avg_processing_time_s):02d}"
            text_position = (20, lane_base_y + 1 * text_spacing)
            font_thickness = 1
            font_face = cv2.FONT_HERSHEY_DUPLEX

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font_face, gui_font, font_thickness
            )
            rectangle_top_left = (
                text_position[0] - 2,
                text_position[1] - text_height - 3,
            )
            rectangle_bottom_right = (
                text_position[0] + text_width + 2,
                text_position[1] + baseline,
            )
            cv2.rectangle(
                frame,
                rectangle_top_left,
                rectangle_bottom_right,
                lane_extraction.lane_colours[lane.id],
                thickness=cv2.FILLED,
            )
            cv2.putText(
                frame,
                text,
                text_position,
                font_face,
                gui_font,
                (255, 255, 255),
                font_thickness,
            )
            cv2.putText(
                frame,
                f"[{lane.id}] CURRENT VEHICLE COUNT {lane.active_vehicle_count}",
                (20, lane_base_y + 2 * text_spacing),
                cv2.FONT_HERSHEY_DUPLEX,
                gui_font,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"[{lane.id}] OUTFLOW COUNT {lane.outflow_count}",
                (20, lane_base_y + 3 * text_spacing),
                cv2.FONT_HERSHEY_DUPLEX,
                gui_font,
                (0, 255, 0),
                1,
            )
    return frame


def main(feed_source, model_name):
    global FRAME_WIDTH, FRAME_HEIGHT
    feed_capture = initialize_feed_capture(feed_source)
    FRAME_WIDTH = int(feed_capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(feed_capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    np.random.seed(15)
    elapsed_time = 0
    vehicles = []
    lane_extraction = LaneExtractor()
    df = create_df()
    while feed_capture.running:
        ret, frame = feed_capture.read()
        if ret:
            elapsed_time = feed_capture.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            # FRAME_WIDTH and FRAME_HEIGHT are already set during initialization
            results = BOUNDING BOXES FROM ONVIF
            vehicles, lane_extraction, df = process_detections(
                results, vehicles, elapsed_time, lane_extraction, df
            )
            if len(lane_extraction.trajectories) >= 40:
                lane_extraction.update_lanes((FRAME_WIDTH, FRAME_HEIGHT))
            vehicles = lane_extraction.assign_vehicles(vehicles, elapsed_time)
            # Add batch update to optimize performance
            vehicles = batch_update_vehicles(vehicles, elapsed_time) 
            frame = draw(frame, vehicles, lane_extraction, elapsed_time)
            cv2.imshow("FEED", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    df.to_csv("data.csv", index=False)
    feed_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        feed_source = sys.argv[1]
        model_name = sys.argv[2]
        main(feed_source, model_name)
    except SystemExit:
        print("Process Terminated")
        pass
