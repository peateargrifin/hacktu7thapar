"""
STEP 1: Bicep Curl Video Analysis
Run this script on your perfect form bicep curl video to extract metrics
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime

def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c where b is the vertex)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_landmark_coords(landmarks, landmark_id):
    """Extract x, y coordinates from landmark"""
    landmark = landmarks.landmark[landmark_id]
    return [landmark.x, landmark.y]

# ============================================================================
# CONFIGURATION - MODIFY THESE
# ============================================================================
VIDEO_PATH = "vids/Denise_Wide-Grip20Curl.mp4"  # <-- CHANGE THIS
USE_LEFT_ARM = True  # Set to False if analyzing right arm
# ============================================================================

# Initialize MediaPipe
mp_pose = mp.solutions.pose

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video file: {VIDEO_PATH}")
    print("Please update VIDEO_PATH in the script to point to your bicep curl video")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate frame interval - process at 15 FPS
if fps >= 15:
    frame_interval = int(fps / 15)
else:
    frame_interval = 1
    
actual_sampling_fps = fps / frame_interval if frame_interval > 0 else fps

print("="*80)
print("BICEP CURL VIDEO ANALYSIS")
print("="*80)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Original FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Frame Interval: {frame_interval}")
print(f"Sampling Rate: ~{actual_sampling_fps:.2f} FPS")
print(f"Analyzing: {'LEFT' if USE_LEFT_ARM else 'RIGHT'} arm")
print("="*80)

# Storage for tracking data
tracking_data = {
    "video_info": {
        "filename": VIDEO_PATH.split('/')[-1],
        "original_fps": fps,
        "sampling_fps": actual_sampling_fps,
        "total_frames_analyzed": 0,
        "arm_analyzed": "LEFT" if USE_LEFT_ARM else "RIGHT",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "frames": [],
    "initial_position": {},
    "rep_metrics": {
        "min_elbow_angle": 180,
        "max_elbow_angle": 0,
        "angle_range": 0,
        "shoulder_stability": [],
        "wrist_alignment": []
    }
}

frame_count = 0
processed_frames = 0

# Select landmarks based on arm choice
if USE_LEFT_ARM:
    SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
    WRIST = mp_pose.PoseLandmark.LEFT_WRIST
    HIP = mp_pose.PoseLandmark.LEFT_HIP
else:
    SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
    ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
    WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
    HIP = mp_pose.PoseLandmark.RIGHT_HIP

# Setup pose model
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only at specified intervals (15 FPS)
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Pose detection
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Get landmark positions
                shoulder = get_landmark_coords(results.pose_landmarks, SHOULDER)
                elbow = get_landmark_coords(results.pose_landmarks, ELBOW)
                wrist = get_landmark_coords(results.pose_landmarks, WRIST)
                hip = get_landmark_coords(results.pose_landmarks, HIP)
                
                # Calculate elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Calculate shoulder angle (shoulder-hip-elbow for posture)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                
                # Store frame data
                frame_data = {
                    "frame_number": frame_count,
                    "time_seconds": round(frame_count / fps, 2),
                    "shoulder": {"x": round(shoulder[0], 4), "y": round(shoulder[1], 4)},
                    "elbow": {"x": round(elbow[0], 4), "y": round(elbow[1], 4)},
                    "wrist": {"x": round(wrist[0], 4), "y": round(wrist[1], 4)},
                    "elbow_angle": round(elbow_angle, 2),
                    "shoulder_angle": round(shoulder_angle, 2)
                }
                
                tracking_data["frames"].append(frame_data)
                
                # Store initial position (first frame)
                if processed_frames == 0:
                    tracking_data["initial_position"] = frame_data.copy()
                    print(f"\n=== INITIAL POSITION (Frame {frame_count}) ===")
                    print(f"Shoulder: {shoulder}")
                    print(f"Elbow: {elbow}")
                    print(f"Wrist: {wrist}")
                    print(f"Elbow Angle: {elbow_angle:.2f}°")
                    print(f"Shoulder Angle: {shoulder_angle:.2f}°\n")
                
                # Track min/max elbow angles
                if elbow_angle < tracking_data["rep_metrics"]["min_elbow_angle"]:
                    tracking_data["rep_metrics"]["min_elbow_angle"] = round(elbow_angle, 2)
                if elbow_angle > tracking_data["rep_metrics"]["max_elbow_angle"]:
                    tracking_data["rep_metrics"]["max_elbow_angle"] = round(elbow_angle, 2)
                
                # Track shoulder stability (movement from initial position)
                shoulder_movement = np.sqrt(
                    (shoulder[0] - tracking_data["initial_position"]["shoulder"]["x"])**2 +
                    (shoulder[1] - tracking_data["initial_position"]["shoulder"]["y"])**2
                )
                tracking_data["rep_metrics"]["shoulder_stability"].append(round(shoulder_movement, 4))
                
                processed_frames += 1
                
                if processed_frames % 15 == 0:  # Print every second
                    print(f"Processed {processed_frames} frames ({frame_count}/{total_frames}) - Elbow Angle: {elbow_angle:.2f}°")

        frame_count += 1

cap.release()

# Calculate final metrics
tracking_data["video_info"]["total_frames_analyzed"] = processed_frames
tracking_data["rep_metrics"]["angle_range"] = round(
    tracking_data["rep_metrics"]["max_elbow_angle"] - tracking_data["rep_metrics"]["min_elbow_angle"], 2
)
tracking_data["rep_metrics"]["avg_shoulder_stability"] = round(
    np.mean(tracking_data["rep_metrics"]["shoulder_stability"]), 4
)
tracking_data["rep_metrics"]["max_shoulder_movement"] = round(
    max(tracking_data["rep_metrics"]["shoulder_stability"]), 4
)

# Find the peak curl point (minimum elbow angle)
peak_curl_frame = min(tracking_data["frames"], key=lambda x: x["elbow_angle"])
tracking_data["rep_metrics"]["peak_curl"] = peak_curl_frame

print("\n" + "="*80)
print("=== ANALYSIS COMPLETE ===")
print("="*80)
print(f"\nTotal Frames Analyzed: {processed_frames}")
print(f"\n*** EXTRACTED METRICS FOR BICEP CURL COUNTER ***")
print("="*80)
print(f"  • Starting Elbow Angle (DOWN position): {tracking_data['frames'][0]['elbow_angle']}°")
print(f"  • Peak Curl Angle (UP position): {tracking_data['rep_metrics']['min_elbow_angle']}°")
print(f"  • Ending Elbow Angle: {tracking_data['frames'][-1]['elbow_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['rep_metrics']['angle_range']}°")
print(f"  • Average Shoulder Stability: {tracking_data['rep_metrics']['avg_shoulder_stability']}")
print(f"  • Max Shoulder Movement: {tracking_data['rep_metrics']['max_shoulder_movement']}")
print("="*80)
print(f"\nPeak Curl occurred at frame {peak_curl_frame['frame_number']} ({peak_curl_frame['time_seconds']}s)")

# Save detailed metrics to text file
with open('bicep_curl_metrics.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BICEP CURL METRICS - USE THESE IN YOUR COUNTER\n")
    f.write("="*80 + "\n\n")
    
    f.write("VIDEO INFO:\n")
    f.write(f"  Filename: {tracking_data['video_info']['filename']}\n")
    f.write(f"  Arm Analyzed: {tracking_data['video_info']['arm_analyzed']}\n")
    f.write(f"  Frames Analyzed: {tracking_data['video_info']['total_frames_analyzed']}\n\n")
    
    f.write("*** THRESHOLDS FOR COUNTER ***\n")
    f.write(f"  DOWN_THRESHOLD (arm extended): {tracking_data['frames'][0]['elbow_angle']}°\n")
    f.write(f"  UP_THRESHOLD (peak curl): {tracking_data['rep_metrics']['min_elbow_angle']}°\n")
    f.write(f"  MIN_RANGE_OF_MOTION: {tracking_data['rep_metrics']['angle_range']}°\n")
    f.write(f"  MAX_SHOULDER_MOVEMENT: {tracking_data['rep_metrics']['max_shoulder_movement']}\n\n")
    
    f.write("INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['initial_position']['shoulder']['x']}, y={tracking_data['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['initial_position']['elbow']['x']}, y={tracking_data['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['initial_position']['wrist']['x']}, y={tracking_data['initial_position']['wrist']['y']}\n")
    f.write(f"  Elbow Angle: {tracking_data['initial_position']['elbow_angle']}°\n\n")
    
    f.write("FRAME-BY-FRAME DATA:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Elbow°':<10} {'Shoulder°':<12} {'Shoulder Drift':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['frames']):
        shoulder_drift = tracking_data['rep_metrics']['shoulder_stability'][i]
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['elbow_angle']:<10} "
                f"{frame['shoulder_angle']:<12} {shoulder_drift:<15}\n")

# Save JSON for programmatic use
with open('bicep_curl_metrics.json', 'w') as f:
    json.dump(tracking_data, f, indent=2)

print(f"\n✓ Metrics saved to 'bicep_curl_metrics.txt'")
print(f"✓ Detailed JSON saved to 'bicep_curl_metrics.json'")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Check 'bicep_curl_metrics.txt' for the extracted thresholds")
print("2. Use these values in step2_improved_bicep_curl_counter.py")
print("3. Update the CURL_METRICS dictionary with your actual values")
print("="*80)