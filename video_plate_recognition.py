"""
License Plate Recognition - Video Processing
This script detects and reads license plates from video files or webcam streams using YOLOv5 and EasyOCR.
"""

import torch
import easyocr
import pathlib
import numpy as np
import cv2
import warnings

# Suppress FutureWarning messages from PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)

# On Windows, unpickling objects that include pathlib.PosixPath (saved on POSIX systems)
# raises: NotImplementedError: cannot instantiate 'PosixPath' on your system.
# Map PosixPath to WindowsPath so torch.load (pickle) can instantiate paths correctly.
pathlib.PosixPath = pathlib.WindowsPath

# Windows path to your custom YOLOv5 weights
weights_path = r"best.pt"

# Load custom YOLOv5 model (force_reload to avoid stale cache)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
print("Model loaded successfully!")

# Initialize EasyOCR reader
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("EasyOCR reader initialized!")

# Process video stream (webcam or video file)

# OPTION 1: Use video file
video_source = r"sample2.mp4"

# OPTION 2: Use webcam (uncomment the line below and comment the line above)
# video_source = 0  # 0 = default webcam, 1 = external webcam

# Open video stream
print(f"\nOpening video source: {video_source}")
cap = cv2.VideoCapture(video_source)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

print("Video stream opened successfully")
print("Press 'q' to quit")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {frame_width}x{frame_height} @ {fps} FPS")

# Optional: Save output video
save_output = True  # Set to False if you don't want to save
if save_output:
    output_video_path = r"output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print(f"Saving output to: {output_video_path}")

frame_count = 0

print("\n" + "="*60)
print("Starting video processing...")
print("="*60 + "\n")

# Process video frame by frame
while True:
    # Read frame
    ret, frame = cap.read()
    
    # Break if no frame is captured (end of video)
    if not ret:
        print("End of video stream or failed to capture frame")
        break
    
    frame_count += 1
    
    # Convert BGR to RGB for YOLOv5
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv5 detection
    results = model(frame_rgb)
    
    # Get detection results
    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    height, width = frame.shape[:2]
    
    # Process each detected plate
    for i, row in enumerate(coordinates):
        confidence = row[4]
        
        if confidence >= 0.3:  # Confidence threshold
            # Get bounding box coordinates
            x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
            
            # Add padding
            padding_x = int((x2 - x1) * 0.05)
            padding_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(width, x2 + padding_x)
            y2 = min(height, y2 + padding_y)
            
            # Crop and resize plate
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size > 0:  # Check if crop is valid
                # Resize to 3x for better OCR
                h, w = plate_crop.shape[:2]
                plate_resized = cv2.resize(plate_crop, (w * 3, h * 3), 
                                           interpolation=cv2.INTER_CUBIC)
                
                # Run OCR
                ocr_result = reader.readtext(plate_resized, 
                                            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract and display text
                if ocr_result:
                    plate_text = ' '.join([r[1] for r in ocr_result])
                    ocr_confidence = max([r[2] for r in ocr_result])
                    
                    # Draw text on frame
                    cv2.putText(frame, plate_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {ocr_confidence:.2f}", (x1, y2+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Print to console (every 30 frames to avoid spam)
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: {plate_text} (conf: {ocr_confidence:.2f})")
                else:
                    cv2.putText(frame, "DETECTING...", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Display frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save frame to output video
    if save_output:
        out.write(frame)
    
    # Display the frame
    cv2.imshow('License Plate Detection - Video Stream', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"Processed {frame_count} frames")
print("Video processing complete!")
print("="*60)
