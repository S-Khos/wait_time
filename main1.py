import asyncio
import av
import cv2
import numpy as np
from onvif import ONVIFCamera
from lxml import etree
from urllib.parse import urlparse, urlunparse, quote_plus
import time
from datetime import datetime
import threading
from typing import Dict, List, Optional
import os
import datetime
import sys
from vehicle import Vehicle
from lane_extractor import LaneExtractor
from util import get_time_format
import pandas as pd
import threading
import queue
import time

# ——— CONFIG ———————————————————————————————————————————————
CAM_IP       = "10.2.0.140"
CAM_PORT     = 80
CAM_USER     = "service"
CAM_PASS     = "Vms-2021!"
# ————————————————————————————————————————————————————————

class BoundingBoxData:
    def __init__(self, obj_id: str, left: float, conf:float, top: float, right: float, bottom: float, timestamp: str=None, obj_class: str=None, overlay: 'MetadataOverlay'=None):
        self.obj_id = obj_id
        # Store original ONVIF coordinates
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.conf = conf
        self.timestamp = timestamp or str(datetime.now())
        self.received_time = time.time()
        self.obj_class = obj_class or "Unknown"
        
        # Transform coordinates immediately if overlay is provided
        if overlay:
            self.orig_x1, self.orig_y1, self.orig_x2, self.orig_y2 = self._transform_coordinates(overlay)
            self.screen_x1, self.screen_y1, self.screen_x2, self.screen_y2 = self._scale_to_screen(overlay)
        else:
            self.orig_x1 = self.orig_y1 = self.orig_x2 = self.orig_y2 = 0
            self.screen_x1 = self.screen_y1 = self.screen_x2 = self.screen_y2 = 0
    def _transform_coordinates(self, overlay):
        """Transform ONVIF coordinates to original frame pixel coordinates."""
        original_center_x = overlay.original_width / 2
        original_center_y = overlay.original_height / 2
        
        # Convert ONVIF coordinates to original frame pixels
        # Note: Flip the Y coordinate by negating it
        orig_x1 = original_center_x + (self.left * original_center_x)
        orig_y1 = original_center_y + (-self.top * original_center_y)  # Flip Y
        orig_x2 = original_center_x + (self.right * original_center_x)
        orig_y2 = original_center_y + (-self.bottom * original_center_y)  # Flip Y
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if orig_x2 < orig_x1:
            orig_x1, orig_x2 = orig_x2, orig_x1
        if orig_y2 < orig_y1:
            orig_y1, orig_y2 = orig_y2, orig_y1
        
        # Clamp coordinates to original frame bounds
        orig_x1 = max(0, min(orig_x1, overlay.original_width - 1))
        orig_y1 = max(0, min(orig_y1, overlay.original_height - 1))
        orig_x2 = max(orig_x1 + 1, min(orig_x2, overlay.original_width))
        orig_y2 = max(orig_y1 + 1, min(orig_y2, overlay.original_height))
            
        return orig_x1, orig_y1, orig_x2, orig_y2    
    def _scale_to_screen(self, overlay):
        """Scale original frame coordinates to screen coordinates."""
        # Calculate separate scaling factors for X and Y (stretch to fit)
        scale_x = overlay.target_width / overlay.original_width
        scale_y = overlay.target_height / overlay.original_height
        
        # Scale coordinates directly without maintaining aspect ratio
        x1 = int(self.orig_x1 * scale_x)
        y1 = int(self.orig_y1 * scale_y)
        x2 = int(self.orig_x2 * scale_x)
        y2 = int(self.orig_y2 * scale_y)
        
        # Clamp coordinates to target screen bounds
        x1 = max(0, min(x1, overlay.target_width - 1))
        y1 = max(0, min(y1, overlay.target_height - 1))
        x2 = max(x1 + 1, min(x2, overlay.target_width))
        y2 = max(y1 + 1, min(y2, overlay.target_height))
        
        return x1, y1, x2, y2

class MetadataOverlay:
    def __init__(self):
        self.latest_boxes: Dict[str, BoundingBoxData] = {}  # Only track latest box per object ID
        self.lock = threading.Lock()
        self.target_width = 1920   # Target screen resolution
        self.target_height = 1080
        self.original_width = 1920  # Will be updated from video stream
        self.original_height = 1080

    def update_object(self, box: BoundingBoxData):
        """Update or add the latest bounding box for an object ID."""
        with self.lock:
            self.latest_boxes[box.obj_id] = box

    def get_active_boxes(self, max_age_seconds: float = 2.0) -> List[BoundingBoxData]:
        """Get all active bounding boxes (not older than max_age_seconds)."""
        with self.lock:
            current_time = time.time()
            active_boxes = []
            expired_ids = []
            
            for obj_id, box in self.latest_boxes.items():
                if current_time - box.received_time <= max_age_seconds:
                    active_boxes.append(box)
                else:
                    expired_ids.append(obj_id)
            
            # Clean up expired objects
            for obj_id in expired_ids:
                del self.latest_boxes[obj_id]
            
            return active_boxes

    def set_original_dimensions(self, width: int, height: int):
        """Set the original video stream dimensions."""
        self.original_width = width
        self.original_height = height

    def set_frame_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height


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
    
    # Unpack the results tuple
    bboxes, cur_track_ids, cur_class_ids = results
    
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

def draw(frame, vehicles, lane_extraction, elapsed_time, overlay=None):
    global FRAME_WIDTH, FRAME_HEIGHT
    
    # Calculate scaling parameters if overlay is provided
    if overlay:
        original_width = overlay.original_width
        original_height = overlay.original_height
        target_width = overlay.target_width
        target_height = overlay.target_height
        
        # Calculate separate scaling factors for X and Y (stretch to fit)
        scale_x = target_width / original_width
        scale_y = target_height / original_height
    else:
        # No scaling applied
        scale_x = 1.0
        scale_y = 1.0

    vehicle_telemetry_font = 0.42
    gui_font = 0.6
    for vehicle in vehicles:        
        if vehicle.yFlow < 1: # not vehicle.processed:
            # Scale vehicle coordinates if overlay is provided
            if overlay:
                scaled_x1 = int(vehicle.x1 * scale_x)
                scaled_y1 = int(vehicle.y1 * scale_y)
                scaled_x2 = int(vehicle.x2 * scale_x)
                scaled_y2 = int(vehicle.y2 * scale_y)
                # Scale centroid
                scaled_centroid_x = int(vehicle.centroid[0] * scale_x)
                scaled_centroid_y = int(vehicle.centroid[1] * scale_y)
                scaled_centroid = (scaled_centroid_x, scaled_centroid_y)
            else:
                scaled_x1, scaled_y1 = vehicle.x1, vehicle.y1
                scaled_x2, scaled_y2 = vehicle.x2, vehicle.y2
                scaled_centroid = vehicle.centroid
            
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
            
            # Draw vehicle as both centroid point and bounding box
            cv2.circle(frame, scaled_centroid, radius=2, color=colour, thickness=-1)
           # cv2.rectangle(frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), colour, 1)
            
            cv2.putText(
                frame,
                f"CLS {vehicle.class_id.upper()}",
                (scaled_x1, scaled_y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"ID {vehicle.id}",
                (scaled_x1, scaled_y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"LN {vehicle.lane_id}",
                (scaled_x1, scaled_y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            wait_time_min, wait_time_s = get_time_format(vehicle.time_waited)
            cv2.putText(
                frame,
                f"WT {int(wait_time_min)}:{int(wait_time_s):02d}",
                (scaled_x1, scaled_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            )
            stop_time_min, stop_time_s = get_time_format(vehicle.get_stop_timer(elapsed_time))
            cv2.putText(
                frame,
                f"STP {int(stop_time_min)}:{int(stop_time_s):02d}",
                (scaled_x1, scaled_y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            ) if int(vehicle.get_stop_timer(elapsed_time)) > 0 else None
            lost_time_min, lost_time_s = get_time_format(vehicle.get_lost_timer(elapsed_time))
            cv2.putText(
                frame,
                f"LST {int(lost_time_min)}:{int(lost_time_s):02d}",
                (scaled_x1, scaled_y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                vehicle_telemetry_font,
                colour,
                2,
            ) if int(vehicle.get_lost_timer(elapsed_time)) > 0 else None
            if vehicle.processing_timer > 0:
                processing_time_min, processing_time_s = get_time_format(
                    vehicle.processing_timer
                )
                cv2.putText(
                    frame,
                    f"OT {int(processing_time_min)}:{int(processing_time_s):02d}",
                    (scaled_x2 - 20, scaled_y1 - 15),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    colour,
                    2,
                )    # Scale lane cluster points if overlay is provided
    if lane_extraction.lanes:
        for id, lane in lane_extraction.lanes.items():
            color = lane_extraction.lane_colours[lane.id]
            for pt in lane.cluster:
                if overlay:
                    scaled_pt_x = int(pt[0] * scale_x)
                    scaled_pt_y = int(pt[1] * scale_y)
                    scaled_pt = (scaled_pt_x, scaled_pt_y)
                else:
                    scaled_pt = tuple(pt)
                cv2.circle(frame, scaled_pt, radius=1, color=color, thickness=-1)

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


def make_auth_rtsp_url(rtsp_url: str, username: str, password: str) -> str:
    pw = quote_plus(password)
    u = urlparse(rtsp_url)
    auth_netloc = f"{username}:{pw}@{u.hostname}"
    if u.port:
        auth_netloc += f":{u.port}"
    return urlunparse((u.scheme, auth_netloc, u.path, u.params, u.query, u.fragment))

def extract_bounding_boxes(xml_root, overlay: 'MetadataOverlay' = None) -> List[BoundingBoxData]:
    """Extract bounding box data from the parsed XML metadata."""
    boxes = []
    
    # Extract timestamp
    timestamp = xml_root.get('UtcTime')
    
    # Handle both tt: and ns0: namespace prefixes
    namespaces = {
        'tt': 'http://www.onvif.org/ver10/schema',
        'ns0': 'http://www.onvif.org/ver10/schema'
    }
    
    # Look for video analytics (object detection) - check both namespace formats
    xml_debug_counter = 0
    
    # First try the original tt: namespace format
    analytics_found = False
    for analytics in xml_root.xpath('.//tt:VideoAnalytics', namespaces=namespaces):
        analytics_found = True
        for frame in analytics.xpath('.//tt:Frame', namespaces=namespaces):
            frame_timestamp = frame.get('UtcTime') or timestamp
            
            # Extract objects
            for obj in frame.xpath('.//tt:Object', namespaces=namespaces):
                box = process_object_element(obj, frame_timestamp, xml_debug_counter, namespaces, 'tt', overlay)
                if box:
                    boxes.append(box)
                    xml_debug_counter += 1
    
    # If no analytics found with tt: namespace, try direct ns0:Object elements (your format)
    if not analytics_found:
        # Look for direct ns0:Object elements in the XML
        for obj in xml_root.xpath('.//ns0:Object', namespaces=namespaces):
            box = process_object_element(obj, timestamp, xml_debug_counter, namespaces, 'ns0', overlay)
            if box:
                boxes.append(box)
                xml_debug_counter += 1
    
    return boxes

def process_object_element(obj, timestamp, debug_counter, namespaces, ns_prefix, overlay=None):
    """Process a single object element and return BoundingBoxData if valid."""
    # Save first few objects to XML files for debugging
    # if debug_counter < 2:
    #     import xml.etree.ElementTree as ET
    #     obj_xml = ET.tostring(obj, encoding='unicode')
    #     with open(f'debug_object_{debug_counter}.xml', 'w', encoding='utf-8') as f:
    #         f.write(obj_xml)
    
    obj_id = obj.get('ObjectId')
    if not obj_id:
        return None
    
    # Extract object class/type based on your XML structure
    obj_class = "Unknown"
    likelihood = 0.0

    for class_elem in obj.xpath('.//tt:Class', namespaces=namespaces):
        for class_candidate in class_elem.xpath('.//tt:ClassCandidate', namespaces=namespaces):
            candidate_type = class_candidate.xpath('.//tt:Type', namespaces=namespaces)
            candidate_likelihood = class_candidate.xpath('.//tt:Likelihood', namespaces=namespaces)
            
            if candidate_type and candidate_type[0].text:
                type_likelihood = 0.0
                if candidate_likelihood and candidate_likelihood[0].text:
                    try:
                        type_likelihood = float(candidate_likelihood[0].text)
                    except (ValueError, TypeError):
                        type_likelihood = 0.0
                
                if type_likelihood >= 0.6:  # 60% or greater
                    obj_class = candidate_type[0].text.strip()
                    likelihood = type_likelihood
                    break

        for extention in class_elem.xpath('.//tt:Extension', namespaces=namespaces):
            for othertypes in extention.xpath('.//tt:OtherTypes', namespaces=namespaces):
                candidate_type = othertypes.xpath('.//tt:Type', namespaces=namespaces)
                candidate_likelihood = othertypes.xpath('.//tt:Likelihood', namespaces=namespaces)
                   
                if candidate_type and candidate_type[0].text:
                    type_likelihood = 0.0
                    if candidate_likelihood and candidate_likelihood[0].text:
                        try:
                            type_likelihood = float(candidate_likelihood[0].text)
                        except (ValueError, TypeError):
                            type_likelihood = 0.0
                    
                    if type_likelihood >= 0.6:  # 60% or greater
                        obj_class = candidate_type[0].text.strip()
                        likelihood = type_likelihood
                        break
        break

    # Skip objects that don't meet the likelihood threshold
    if likelihood < 0.6:
        print(f"⚠️ Skipping object {obj_id} - likelihood {likelihood:.2f} < 0.5")
        return None
    
    # Extract bounding box - works for both formats
    bbox_xpath = f'.//{ns_prefix}:BoundingBox'
    for bbox in obj.xpath(bbox_xpath, namespaces=namespaces):
        try:
            left = float(bbox.get('left', 0))
            top = float(bbox.get('top', 0))
            right = float(bbox.get('right', 0))
            bottom = float(bbox.get('bottom', 0))
            
            box = BoundingBoxData(obj_id, left, likelihood, top, right, bottom, timestamp, obj_class, overlay)
            print(f"📦 Object {obj_id} ({obj_class}, {likelihood:.2f}): ONVIF=({left:.3f},{top:.3f},{right:.3f},{bottom:.3f}) -> Original=({box.orig_x1:.1f},{box.orig_y1:.1f},{box.orig_x2:.1f},{box.orig_y2:.1f})")
            return box
            
        except (ValueError, TypeError) as e:
            print(f"⚠️ Error parsing coordinates for object {obj_id}: {e}")
    
    return None

def parse_metadata_stream(uri: str, overlay: MetadataOverlay):
    """Parse metadata stream and extract bounding boxes."""
    print("▶ Opening metadata stream…")
    
    options = {
        'rtsp_transport': 'tcp',
        'stimeout': '5000000',
        'max_delay': '500000',
    }
    
    try:
        container = av.open(uri, options=options)
        
        # Find the data stream (metadata)
        data_stream = None
        for stream in container.streams:
            if stream.type == 'data':
                data_stream = stream
                break
        
        if not data_stream:
            print("❌ No data stream found for metadata")
            return
        
        print(f"🎯 Using data stream: {data_stream}")
        
        buffer = b""        
        for packet in container.demux():
            if packet.stream_index != data_stream.index:
                continue
            
            try:
                data = bytes(packet)
            except Exception:
                continue
            
            if not data:
                continue
            
            buffer += data
            
            # Look for complete XML documents - handle both formats
            patterns = [
                (b"<tt:MetadataStream", b"</tt:MetadataStream>"),
                (b"<ns0:Object", b"</ns0:Object>"),
                (b"<MetadataStream", b"</MetadataStream>"),
            ]
            
            for start_pattern, end_pattern in patterns:
                while start_pattern in buffer:
                    start_idx = buffer.find(start_pattern)
                    end_idx = buffer.find(end_pattern, start_idx)
                    
                    if end_idx == -1:
                        break
                    
                    xml_end = end_idx + len(end_pattern)
                    xml_data = buffer[start_idx:xml_end]
                    buffer = buffer[xml_end:]
                    
                    try:
                        root = etree.fromstring(xml_data)
                        bounding_boxes = extract_bounding_boxes(root, overlay)
                        
                        # Update each object's latest position
                        for box in bounding_boxes:
                            overlay.update_object(box)
                    
                    except Exception as e:
                        print(f"❌ XML parsing error: {e}")
                        # Save problematic XML for debugging
                        with open('debug_error.xml', 'wb') as f:
                            f.write(xml_data)
        
    except Exception as e:
        print(f"❌ Error opening metadata stream: {e}")

def scale_frame_to_screen(frame: np.ndarray, target_width: int = 1920, target_height: int = 1080) -> np.ndarray:
    """Scale frame to target screen resolution by stretching to fill the entire screen."""
    original_height, original_width = frame.shape[:2]
    
    # Resize the frame to exactly match target resolution (stretch to fit)
    scaled_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    return scaled_frame

def video_stream_with_overlay(video_uri: str, overlay: MetadataOverlay):
    """Display video stream with bounding box overlay scaled to screen resolution."""
    print("🎥 Opening video stream...")

    global FRAME_WIDTH, FRAME_HEIGHT
    
    cap = cv2.VideoCapture(video_uri)
    if not cap.isOpened():
        print("❌ Failed to open video stream")
        return
    
    # Get original frame dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    FRAME_WIDTH = original_width
    FRAME_HEIGHT = original_height
    
    overlay.set_original_dimensions(original_width, original_height)
    
    print(f"📺 Original Stream: {original_width}x{original_height} @ {fps:.1f}fps")
    print(f"🖥️ Scaling to: {overlay.target_width}x{overlay.target_height}")
    
    # Initialize tracking variables
    np.random.seed(15)
    elapsed_time = 0
    vehicles = []
    lane_extraction = LaneExtractor()
    df = create_df()
    start_time = time.time()
    
    frame_count = 0
    cv2.namedWindow('RTSP Stream with Bounding Boxes', cv2.WINDOW_AUTOSIZE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video stream")
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Scale frame to screen resolution
        scaled_frame = scale_frame_to_screen(frame, overlay.target_width, overlay.target_height)
        
        # Get active bounding boxes (latest for each object)
        active_boxes = overlay.get_active_boxes()
        
        if active_boxes:
            # Convert bounding boxes from BoundingBoxData objects to the format expected by process_detections
            # Use pre-transformed original frame coordinates for vehicle tracking
            bboxes = []
            cur_track_ids = []
            cur_class_ids = []
            
            for box in active_boxes:
                # Use pre-transformed original frame coordinates
                x = box.orig_x1
                y = box.orig_y1
                w = box.orig_x2 - box.orig_x1
                h = box.orig_y2 - box.orig_y1
                
                bboxes.append([x, y, w, h])
                cur_track_ids.append(int(box.obj_id))
                cur_class_ids.append(box.obj_class.lower())
            
            # Convert to numpy arrays
            bboxes = np.array(bboxes)
            cur_track_ids = np.array(cur_track_ids)
            cur_class_ids = np.array(cur_class_ids)

            vehicles, lane_extraction, df = process_detections(
                (bboxes, cur_track_ids, cur_class_ids), vehicles, elapsed_time, lane_extraction, df
            )
            if len(lane_extraction.trajectories) >= 40:
                lane_extraction.update_lanes((FRAME_WIDTH, FRAME_HEIGHT))
            vehicles = lane_extraction.assign_vehicles(vehicles, elapsed_time)
            # Add batch update to optimize performance
            vehicles = batch_update_vehicles(vehicles, elapsed_time) 
        
        # Draw vehicles and GUI elements (no longer passing active_boxes)
        scaled_frame = draw(scaled_frame, vehicles, lane_extraction, elapsed_time, overlay)
        
        cv2.imshow('RTSP Stream with Bounding Boxes', scaled_frame)
        
        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
async def main():
    try:
        print("🔌 Connecting to camera...")
        camera = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
        await camera.update_xaddrs()
        
        print("🔍 Getting media profiles...")
        media = await camera.create_media_service()
        profiles = await media.GetProfiles()
        
        # Get both video and metadata URIs
        video_profile = profiles[0]  # Assuming first profile has video
        metadata_profile = profiles[0]  # Same profile for metadata
        
        print(f"🎯 Selected profile: {video_profile.Name}")

        # Get video stream URI
        print("🎥 Getting video stream URI...")
        uri_req = media.create_type('GetStreamUri')
        uri_req.ProfileToken = video_profile.token
        uri_req.StreamSetup = {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        }
        uri_resp = await media.GetStreamUri(uri_req)
        video_uri = make_auth_rtsp_url(uri_resp.Uri, CAM_USER, CAM_PASS)
        
        # Get metadata stream URI
        print("📊 Getting metadata stream URI...")
        uri_req.ProfileToken = metadata_profile.token
        uri_resp = await media.GetStreamUri(uri_req)
        metadata_uri = make_auth_rtsp_url(uri_resp.Uri, CAM_USER, CAM_PASS)
        
        print(f"▶ Video URI: {video_uri}")
        print(f"▶ Metadata URI: {metadata_uri}")

        # Create overlay manager
        overlay = MetadataOverlay()
        
        # Start metadata parsing in background thread
        print("🚀 Starting metadata stream parsing...")
        metadata_thread = threading.Thread(
            target=parse_metadata_stream, 
            args=(metadata_uri, overlay),
            daemon=True
        )
        metadata_thread.start()
        
        # Give metadata stream time to start
        await asyncio.sleep(2)
        
        # Start video stream with overlay (this runs in main thread)
        print("🎬 Starting video display...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, video_stream_with_overlay, video_uri, overlay)
        
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 RTSP Video Stream with ONVIF Metadata Overlay")
    print("Press 'q' to quit the video display")
    print("=" * 60)
    asyncio.run(main())