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

# ——— CONFIG ———————————————————————————————————————————————
CAM_IP       = "10.2.0.140"
CAM_PORT     = 80
CAM_USER     = "service"
CAM_PASS     = "Vms-2021!"
# ————————————————————————————————————————————————————————

class BoundingBoxData:
    def __init__(self, obj_id: str, left: float, top: float, right: float, bottom: float, timestamp: str=None):
        self.obj_id = obj_id
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.timestamp = timestamp or str(datetime.now())
        self.received_time = time.time()

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

def make_auth_rtsp_url(rtsp_url: str, username: str, password: str) -> str:
    pw = quote_plus(password)
    u = urlparse(rtsp_url)
    auth_netloc = f"{username}:{pw}@{u.hostname}"
    if u.port:
        auth_netloc += f":{u.port}"
    return urlunparse((u.scheme, auth_netloc, u.path, u.params, u.query, u.fragment))

def extract_bounding_boxes(xml_root) -> List[BoundingBoxData]:
    """Extract bounding box data from the parsed XML metadata."""
    boxes = []
    
    # Extract timestamp
    timestamp = xml_root.get('UtcTime')
    
    # Look for video analytics (object detection)
    for analytics in xml_root.xpath('.//tt:VideoAnalytics', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
        for frame in analytics.xpath('.//tt:Frame', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
            frame_timestamp = frame.get('UtcTime') or timestamp
            
            # Extract objects
            for obj in frame.xpath('.//tt:Object', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                obj_id = obj.get('ObjectId')
                if not obj_id:
                    continue
                
                # Extract bounding box
                for bbox in obj.xpath('.//tt:BoundingBox', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                    try:
                        left = float(bbox.get('left', 0))
                        top = float(bbox.get('top', 0))
                        right = float(bbox.get('right', 0))
                        bottom = float(bbox.get('bottom', 0))
                        
                        box = BoundingBoxData(obj_id, left, top, right, bottom, frame_timestamp)
                        boxes.append(box)
                        
                        print(f"📦 Object {obj_id}: ({left:.3f},{top:.3f}) to ({right:.3f},{bottom:.3f})")
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Error parsing coordinates for object {obj_id}: {e}")
    
    return boxes

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
            
            # Look for complete XML documents
            start_pattern = b"<tt:MetadataStream"
            end_pattern = b"</tt:MetadataStream>"
            
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
                    bounding_boxes = extract_bounding_boxes(root)
                    
                    # Update each object's latest position
                    for box in bounding_boxes:
                        overlay.update_object(box)
                
                except Exception as e:
                    print(f"❌ XML parsing error: {e}")
        
    except Exception as e:
        print(f"❌ Error opening metadata stream: {e}")

def draw_bounding_boxes(frame: np.ndarray, boxes: List[BoundingBoxData], overlay: MetadataOverlay) -> np.ndarray:
    """Draw bounding boxes on the scaled frame with proper coordinate conversion."""
    height, width = frame.shape[:2]  # This should be 1080x1920 after scaling
    
    # Calculate scaling info to properly map coordinates
    original_width = overlay.original_width
    original_height = overlay.original_height
    target_width = overlay.target_width
    target_height = overlay.target_height
    
    # Calculate the actual scaling used
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)  # Same logic as in scale_frame_to_screen
    
    # Calculate the actual scaled dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Calculate offsets for centering (padding)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    for box in boxes:
        # ONVIF uses center-based coordinate system: (-1,-1) to (+1,+1)
        # Center of original frame is (0,0)
        # BUT: Y-axis is flipped in ONVIF (bottom = -1, top = +1) vs OpenCV (top = 0, bottom = height)
        
        # Convert normalized coordinates to pixels in the original frame space
        original_center_x = original_width / 2
        original_center_y = original_height / 2
        
        # Convert ONVIF coordinates to original frame pixels
        # Note: Flip the Y coordinate by negating it
        orig_x1 = original_center_x + (box.left * original_center_x)
        orig_y1 = original_center_y + (-box.top * original_center_y)  # Flip Y
        orig_x2 = original_center_x + (box.right * original_center_x)
        orig_y2 = original_center_y + (-box.bottom * original_center_y)  # Flip Y
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if orig_x2 < orig_x1:
            orig_x1, orig_x2 = orig_x2, orig_x1
        if orig_y2 < orig_y1:
            orig_y1, orig_y2 = orig_y2, orig_y1
        
        # Scale coordinates to the scaled frame and add padding offset
        x1 = int(orig_x1 * scale) + x_offset
        y1 = int(orig_y1 * scale) + y_offset
        x2 = int(orig_x2 * scale) + x_offset
        y2 = int(orig_y2 * scale) + y_offset
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # Skip invalid boxes
        if x1 >= x2 or y1 >= y2:
            continue
        
        print(f"🔍 Box {box.obj_id}: ONVIF=({box.left:.3f},{box.top:.3f},{box.right:.3f},{box.bottom:.3f}) -> Original=({orig_x1:.1f},{orig_y1:.1f},{orig_x2:.1f},{orig_y2:.1f}) -> Scaled=({x1},{y1},{x2},{y2})")
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw object ID label
        label = f"ID: {box.obj_id}"
        cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame

def scale_frame_to_screen(frame: np.ndarray, target_width: int = 1920, target_height: int = 1080) -> np.ndarray:
    """Scale frame to target screen resolution while maintaining aspect ratio."""
    original_height, original_width = frame.shape[:2]
    
    # Calculate scaling factor to fit within target resolution
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)  # Use the smaller scale to maintain aspect ratio
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the frame
    scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # If the scaled frame doesn't exactly match target resolution, pad with black borders
    if new_width != target_width or new_height != target_height:
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the scaled frame
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place the scaled frame on the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_frame
        return canvas
    
    return scaled_frame

def video_stream_with_overlay(video_uri: str, overlay: MetadataOverlay):
    """Display video stream with bounding box overlay scaled to screen resolution."""
    print("🎥 Opening video stream...")
    
    cap = cv2.VideoCapture(video_uri)
    if not cap.isOpened():
        print("❌ Failed to open video stream")
        return
    
    # Get original frame dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    overlay.set_original_dimensions(original_width, original_height)
    
    print(f"📺 Original Stream: {original_width}x{original_height} @ {fps:.1f}fps")
    print(f"🖥️ Scaling to: {overlay.target_width}x{overlay.target_height}")
    
    frame_count = 0
    cv2.namedWindow('RTSP Stream with Bounding Boxes', cv2.WINDOW_AUTOSIZE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Scale frame to screen resolution
        scaled_frame = scale_frame_to_screen(frame, overlay.target_width, overlay.target_height)
        
        # Get active bounding boxes (latest for each object)
        active_boxes = overlay.get_active_boxes()
        
        # Draw bounding boxes on scaled frame
        if active_boxes:
            scaled_frame = draw_bounding_boxes(scaled_frame, active_boxes, overlay)
        
        # Add frame info
        # #info_text = f"Frame: {frame_count} | Active Objects: {len(active_boxes)} | {overlay.target_width}x{overlay.target_height}"
        # cv2.putText(scaled_frame, info_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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