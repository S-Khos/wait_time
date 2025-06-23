import asyncio
import av
from onvif import ONVIFCamera
from lxml import etree
from urllib.parse import urlparse, urlunparse, quote_plus
import time
from datetime import datetime

# ——— CONFIG ———————————————————————————————————————————————
CAM_IP       = "10.2.0.140"
CAM_PORT     = 80
CAM_USER     = "service"
CAM_PASS     = "Vms-2021!"
# ————————————————————————————————————————————————————————

def make_auth_rtsp_url(rtsp_url: str, username: str, password: str) -> str:
    pw = quote_plus(password)
    u = urlparse(rtsp_url)
    auth_netloc = f"{username}:{pw}@{u.hostname}"
    if u.port:
        auth_netloc += f":{u.port}"
    return urlunparse((u.scheme, auth_netloc, u.path, u.params, u.query, u.fragment))

def extract_analytics_data(xml_root):
    """
    Extract meaningful analytics data from the parsed XML metadata.
    """
    data = {
        'timestamp': None,
        'events': [],
        'objects': [],
        'regions': [],
        'raw_points': []
    }
    
    # Extract timestamp
    try:
        utc_time = xml_root.get('UtcTime')
        if utc_time:
            data['timestamp'] = utc_time
    except:
        pass
    
    # Look for events
    for event in xml_root.xpath('.//tt:Event', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
        try:
            # Extract notification messages
            for notification in event.xpath('.//wsnt:NotificationMessage', 
                                          namespaces={'wsnt': 'http://docs.oasis-open.org/wsn/b-2'}):
                event_data = {}
                
                # Get topic
                topic_elem = notification.xpath('.//wsnt:Topic', 
                                              namespaces={'wsnt': 'http://docs.oasis-open.org/wsn/b-2'})
                if topic_elem:
                    event_data['topic'] = topic_elem[0].text
                
                # Get message data
                message_elem = notification.xpath('.//wsnt:Message', 
                                                namespaces={'wsnt': 'http://docs.oasis-open.org/wsn/b-2'})
                if message_elem:
                    # Extract simple items
                    for item in message_elem[0].xpath('.//tt:SimpleItem', 
                                                    namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                        name = item.get('Name')
                        value = item.get('Value')
                        if name and value:
                            event_data[name] = value
                
                if event_data:
                    data['events'].append(event_data)
        except Exception as e:
            print(f"⚠️ Error parsing event: {e}")
    
    # Look for video analytics (object detection, etc.)
    for analytics in xml_root.xpath('.//tt:VideoAnalytics', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
        try:
            # Extract frames
            for frame in analytics.xpath('.//tt:Frame', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                frame_data = {
                    'timestamp': frame.get('UtcTime'),
                    'objects': []
                }
                
                # Extract objects
                for obj in frame.xpath('.//tt:Object', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                    obj_data = {
                        'id': obj.get('ObjectId'),
                        'appearance': {},
                        'geometry': {}
                    }
                    
                    # Extract appearance
                    for appearance in obj.xpath('.//tt:Appearance', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                        for shape in appearance.xpath('.//tt:Shape', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                            # Extract bounding box or polygon
                            bbox = shape.xpath('.//tt:BoundingBox', namespaces={'tt': 'http://www.onvif.org/ver10/schema'})
                            if bbox:
                                obj_data['geometry']['type'] = 'BoundingBox'
                                obj_data['geometry']['left'] = float(bbox[0].get('left', 0))
                                obj_data['geometry']['top'] = float(bbox[0].get('top', 0))
                                obj_data['geometry']['right'] = float(bbox[0].get('right', 0))
                                obj_data['geometry']['bottom'] = float(bbox[0].get('bottom', 0))
                            
                            # Extract polygon points
                            for polygon in shape.xpath('.//tt:Polygon', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                                points = []
                                for point in polygon.xpath('.//tt:Point', namespaces={'tt': 'http://www.onvif.org/ver10/schema'}):
                                    x = float(point.get('x', 0))
                                    y = float(point.get('y', 0))
                                    points.append((x, y))
                                    data['raw_points'].append((x, y))  # Also store raw points
                                
                                if points:
                                    obj_data['geometry']['type'] = 'Polygon'
                                    obj_data['geometry']['points'] = points
                    
                    if obj_data['id']:
                        frame_data['objects'].append(obj_data)
                
                if frame_data['objects']:
                    data['objects'].append(frame_data)
        except Exception as e:
            print(f"⚠️ Error parsing video analytics: {e}")
    
    return data

def parse_metadata_stream(uri: str):
    """
    Opens the RTSP metadata stream and continuously parses XML metadata.
    """
    print("▶ Opening metadata stream…")
    
    options = {
        'rtsp_transport': 'tcp',
        'stimeout': '5000000',
        'max_delay': '500000',
    }
    
    try:
        container = av.open(uri, options=options)
        print(f"Container: {container}")
        print(f"Streams: {len(container.streams)}")
        
        # Find the data stream (metadata)
        data_stream = None
        for i, stream in enumerate(container.streams):
            print(f"Stream {i}: {stream} (Type: {stream.type})")
            if stream.type == 'data':
                data_stream = stream
                break
        
        if not data_stream:
            print("❌ No data stream found for metadata")
            return
        
        print(f"🎯 Using data stream: {data_stream}")
        
        buffer = b""
        packet_count = 0
        metadata_count = 0
        
        print("\n" + "="*80)
        print("🚀 STARTING METADATA STREAM PROCESSING")
        print("="*80)
        
        for packet in container.demux():
            # Only process data stream packets (metadata)
            if packet.stream_index != data_stream.index:
                continue
                
            packet_count += 1
            
            try:
                data = bytes(packet)
            except Exception as e:
                print(f"⚠️ Failed to extract packet data: {e}")
                continue
            
            if not data:
                continue
            
            buffer += data
            
            # Look for complete XML documents
            # The pattern from your output shows XML starts with <tt:MetadataStream and ends with </tt:MetadataStream>
            start_pattern = b"<tt:MetadataStream"
            end_pattern = b"</tt:MetadataStream>"
            
            while start_pattern in buffer:
                start_idx = buffer.find(start_pattern)
                end_idx = buffer.find(end_pattern, start_idx)
                
                if end_idx == -1:
                    # Incomplete XML, wait for more data
                    break
                
                # Extract complete XML document
                xml_end = end_idx + len(end_pattern)
                xml_data = buffer[start_idx:xml_end]
                buffer = buffer[xml_end:]  # Remove processed XML from buffer
            
                
                try:
                    # Parse XML
                    xml_text = xml_data.decode('utf-8', errors='ignore')
                    root = etree.fromstring(xml_data)
                    
                    # Extract analytics data
                    analytics = extract_analytics_data(root)
                    
                    # Display parsed data
                    if analytics['timestamp']:
                        print(f"🕐 Timestamp: {analytics['timestamp']}")
                    
                    if analytics['events']:
                        print(f"🚨 Events ({len(analytics['events'])}):")
                        for i, event in enumerate(analytics['events']):
                            print(f"  Event {i+1}: {event}")
                    
                    if analytics['objects']:
                        print(f"👁️  Objects Detected ({len(analytics['objects'])}):")
                        for frame in analytics['objects']:
                            if frame['timestamp']:
                                print(f"  Frame at {frame['timestamp']}:")
                            for j, obj in enumerate(frame['objects']):
                                print(f"    Object {j+1} (ID: {obj['id']}):")
                                if obj['geometry']:
                                    if obj['geometry']['type'] == 'BoundingBox':
                                        geo = obj['geometry']
                                        print(f"      BoundingBox: ({geo['left']:.3f}, {geo['top']:.3f}) to ({geo['right']:.3f}, {geo['bottom']:.3f})")
                                    elif obj['geometry']['type'] == 'Polygon':
                                        points = obj['geometry']['points']
                                        print(f"      Polygon: {len(points)} points")
                                        for k, (x, y) in enumerate(points[:5]):  # Show first 5 points
                                            print(f"        Point {k+1}: ({x:.3f}, {y:.3f})")
                                        if len(points) > 5:
                                            print(f"        ... and {len(points)-5} more points")
                    
                    if analytics['raw_points']:
                        print(f"📍 Raw Points ({len(analytics['raw_points'])}):")
                        # Group points and show summary
                        unique_points = list(set(analytics['raw_points']))
                        if len(unique_points) <= 10:
                            for x, y in unique_points:
                                print(f"    ({x:.3f}, {y:.3f})")
                        else:
                            print(f"    First 5: {unique_points[:5]}")
                            print(f"    ... and {len(unique_points)-5} more unique points")
                    
                    # Show raw XML for first few metadata packets (for debugging)
                    if metadata_count <= 3:
                        print(f"\n📄 Raw XML (first {min(500, len(xml_text))} chars):")
                        print(xml_text[:500] + ("..." if len(xml_text) > 500 else ""))
                
                except Exception as e:
                    print(f"❌ XML parsing error: {e}")
                    print(f"📄 Raw data preview:")
                    preview = xml_data[:200].decode('utf-8', errors='ignore')
                    print(f"  {preview}...")
                
                # # Limit output for demonstration
                # if metadata_count >= 20:
                #     print(f"\n🛑 Stopping after {metadata_count} metadata packets for demo")
                #     print("   (Remove this limit to run continuously)")
                #     break
        
        print(f"\n✅ Processed {packet_count} packets, found {metadata_count} metadata documents")
        
    except Exception as e:
        print(f"❌ Error opening stream: {e}")
        import traceback
        traceback.print_exc()

async def main():
    try:
        print("🔌 Connecting to camera...")
        camera = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
        await camera.update_xaddrs()
        
        print("🔍 Getting media profiles...")
        media = await camera.create_media_service()
        profiles = await media.GetProfiles()
        
        # Find metadata profile (we know from your output that Profile_L1S1 works)
        metadata_profile = profiles[0]  # Use first profile which has metadata
        print(f"🎯 Selected profile: {metadata_profile.Name} (Token: {metadata_profile.token})")

        print("🎥 Getting stream URI...")
        uri_req = media.create_type('GetStreamUri')
        uri_req.ProfileToken = metadata_profile.token
        uri_req.StreamSetup = {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        }
        uri_resp = await media.GetStreamUri(uri_req)
        metadata_uri = uri_resp.Uri

        auth_uri = make_auth_rtsp_url(metadata_uri, CAM_USER, CAM_PASS)
        print(f"▶ Metadata URI: {auth_uri}")

        print("🚀 Starting metadata stream parsing...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, parse_metadata_stream, auth_uri)
        
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())