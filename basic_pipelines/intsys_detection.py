import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import base64
import requests
import time
import asyncio
import threading
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import fractions

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.webrtc_connected = False
        self.video_track = None

    def new_function(self):  # New function example
        return "The meaning of life is: "
    
    def set_video_track(self, track):
        self.video_track = track
        self.webrtc_connected = True

# Custom VideoStreamTrack for WebRTC
class DetectionVideoStreamTrack(VideoStreamTrack):
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.frame = None
        self.pts = 0
        self.time_base = fractions.Fraction(1, 90000)
        # Default black frame
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def push_frame(self, frame):
        self.frame = frame
    
    async def recv(self):
        if self.frame is None:
            # Return black frame if no frame is available
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img = self.frame
            
        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(img, format="bgr24")
        video_frame.pts = self.pts
        video_frame.time_base = self.time_base
        self.pts += 3000  # 30fps
        
        return video_frame

# -----------------------------------------------------------------------------------------------
# WebRTC Connection Management
# -----------------------------------------------------------------------------------------------

# Server configuration
SERVER_URL = "http://10.48.61.73:8888"

# Function to establish WebRTC connection with server
async def setup_webrtc_connection(user_data):
    # Create peer connection
    pc = RTCPeerConnection()
    
    # Create video track
    video_track = DetectionVideoStreamTrack()
    user_data.set_video_track(video_track)
    
    # Add track to peer connection
    pc.addTrack(video_track)
    
    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Send offer to server
    response = requests.post(
        f"{SERVER_URL}/raspberry-pi/offer",
        json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )
    
    if response.status_code != 200:
        print(f"Failed to send offer: {response.status_code}")
        return False
    
    # Get answer from server
    answer = response.json()
    
    # Set remote description
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    
    print("WebRTC connection established!")
    return True

# Function to run the asyncio event loop
def run_asyncio_loop(user_data):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(setup_webrtc_connection(user_data))
    loop.run_forever()

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    print(user_data.use_frame, format, width, height)
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
            
            # Draw bounding box if we have a frame
            if frame is not None:
                # Get bbox coordinates (normalized)
                x1_norm = bbox.xmin()
                y1_norm = bbox.ymin() 
                x2_norm = bbox.xmax()
                y2_norm = bbox.ymax()
                
                # Convert to pixel coordinates
                height, width = frame.shape[:2]
                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                x2 = int(x2_norm * width)
                y2 = int(y2_norm * height)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if user_data.use_frame and frame is not None:
        # Add detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
        
        # Send the frame via WebRTC if connected
        if user_data.webrtc_connected and user_data.video_track:
            user_data.video_track.push_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    user_data.use_frame = True  # Make sure we get frames

    # Start WebRTC connection in a separate thread
    webrtc_thread = threading.Thread(target=run_asyncio_loop, args=(user_data,))
    webrtc_thread.daemon = True
    webrtc_thread.start()

    # Run the detection app
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
