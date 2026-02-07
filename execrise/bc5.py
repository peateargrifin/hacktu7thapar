"""
STEP 1: Bicep Curl Video Analysis - BOTH ARMS
Run this script on your perfect form bicep curl video to extract metrics for BOTH arms
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
print("BICEP CURL VIDEO ANALYSIS - BOTH ARMS")
print("="*80)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Original FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Frame Interval: {frame_interval}")
print(f"Sampling Rate: ~{actual_sampling_fps:.2f} FPS")
print(f"Analyzing: BOTH LEFT and RIGHT arms")
print("="*80)

# Storage for tracking data - separate for each arm
tracking_data = {
    "video_info": {
        "filename": VIDEO_PATH.split('/')[-1],
        "original_fps": fps,
        "sampling_fps": actual_sampling_fps,
        "total_frames_analyzed": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "left_arm": {
        "frames": [],
        "initial_position": {},
        "rep_metrics": {
            "min_elbow_angle": 180,
            "max_elbow_angle": 0,
            "angle_range": 0,
            "shoulder_stability": [],
            "wrist_alignment": []
        }
    },
    "right_arm": {
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
}

frame_count = 0
processed_frames = 0

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
                # ============================================================
                # PROCESS LEFT ARM
                # ============================================================
                left_shoulder = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                left_elbow = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
                left_wrist = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
                left_hip = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                
                # Calculate left arm angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                
                # Store left arm frame data
                left_frame_data = {
                    "frame_number": frame_count,
                    "time_seconds": round(frame_count / fps, 2),
                    "shoulder": {"x": round(left_shoulder[0], 4), "y": round(left_shoulder[1], 4)},
                    "elbow": {"x": round(left_elbow[0], 4), "y": round(left_elbow[1], 4)},
                    "wrist": {"x": round(left_wrist[0], 4), "y": round(left_wrist[1], 4)},
                    "elbow_angle": round(left_elbow_angle, 2),
                    "shoulder_angle": round(left_shoulder_angle, 2)
                }
                
                tracking_data["left_arm"]["frames"].append(left_frame_data)
                
                # Store initial position for left arm (first frame)
                if processed_frames == 0:
                    tracking_data["left_arm"]["initial_position"] = left_frame_data.copy()
                    print(f"\n=== LEFT ARM - INITIAL POSITION (Frame {frame_count}) ===")
                    print(f"Shoulder: {left_shoulder}")
                    print(f"Elbow: {left_elbow}")
                    print(f"Wrist: {left_wrist}")
                    print(f"Elbow Angle: {left_elbow_angle:.2f}°")
                    print(f"Shoulder Angle: {left_shoulder_angle:.2f}°")
                
                # Track min/max elbow angles for left arm
                if left_elbow_angle < tracking_data["left_arm"]["rep_metrics"]["min_elbow_angle"]:
                    tracking_data["left_arm"]["rep_metrics"]["min_elbow_angle"] = round(left_elbow_angle, 2)
                if left_elbow_angle > tracking_data["left_arm"]["rep_metrics"]["max_elbow_angle"]:
                    tracking_data["left_arm"]["rep_metrics"]["max_elbow_angle"] = round(left_elbow_angle, 2)
                
                # Track left shoulder stability
                if processed_frames > 0:
                    left_shoulder_movement = np.sqrt(
                        (left_shoulder[0] - tracking_data["left_arm"]["initial_position"]["shoulder"]["x"])**2 +
                        (left_shoulder[1] - tracking_data["left_arm"]["initial_position"]["shoulder"]["y"])**2
                    )
                    tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"].append(round(left_shoulder_movement, 4))
                
                # ============================================================
                # PROCESS RIGHT ARM
                # ============================================================
                right_shoulder = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                right_elbow = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
                right_wrist = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
                right_hip = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                
                # Calculate right arm angles
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                
                # Store right arm frame data
                right_frame_data = {
                    "frame_number": frame_count,
                    "time_seconds": round(frame_count / fps, 2),
                    "shoulder": {"x": round(right_shoulder[0], 4), "y": round(right_shoulder[1], 4)},
                    "elbow": {"x": round(right_elbow[0], 4), "y": round(right_elbow[1], 4)},
                    "wrist": {"x": round(right_wrist[0], 4), "y": round(right_wrist[1], 4)},
                    "elbow_angle": round(right_elbow_angle, 2),
                    "shoulder_angle": round(right_shoulder_angle, 2)
                }
                
                tracking_data["right_arm"]["frames"].append(right_frame_data)
                
                # Store initial position for right arm (first frame)
                if processed_frames == 0:
                    tracking_data["right_arm"]["initial_position"] = right_frame_data.copy()
                    print(f"\n=== RIGHT ARM - INITIAL POSITION (Frame {frame_count}) ===")
                    print(f"Shoulder: {right_shoulder}")
                    print(f"Elbow: {right_elbow}")
                    print(f"Wrist: {right_wrist}")
                    print(f"Elbow Angle: {right_elbow_angle:.2f}°")
                    print(f"Shoulder Angle: {right_shoulder_angle:.2f}°\n")
                
                # Track min/max elbow angles for right arm
                if right_elbow_angle < tracking_data["right_arm"]["rep_metrics"]["min_elbow_angle"]:
                    tracking_data["right_arm"]["rep_metrics"]["min_elbow_angle"] = round(right_elbow_angle, 2)
                if right_elbow_angle > tracking_data["right_arm"]["rep_metrics"]["max_elbow_angle"]:
                    tracking_data["right_arm"]["rep_metrics"]["max_elbow_angle"] = round(right_elbow_angle, 2)
                
                # Track right shoulder stability
                if processed_frames > 0:
                    right_shoulder_movement = np.sqrt(
                        (right_shoulder[0] - tracking_data["right_arm"]["initial_position"]["shoulder"]["x"])**2 +
                        (right_shoulder[1] - tracking_data["right_arm"]["initial_position"]["shoulder"]["y"])**2
                    )
                    tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"].append(round(right_shoulder_movement, 4))
                
                processed_frames += 1
                
                if processed_frames % 15 == 0:  # Print every second
                    print(f"Processed {processed_frames} frames ({frame_count}/{total_frames}) - "
                          f"Left: {left_elbow_angle:.2f}° | Right: {right_elbow_angle:.2f}°")

        frame_count += 1

cap.release()

# ============================================================================
# Calculate final metrics for BOTH arms
# ============================================================================
tracking_data["video_info"]["total_frames_analyzed"] = processed_frames

# LEFT ARM METRICS
tracking_data["left_arm"]["rep_metrics"]["angle_range"] = round(
    tracking_data["left_arm"]["rep_metrics"]["max_elbow_angle"] - 
    tracking_data["left_arm"]["rep_metrics"]["min_elbow_angle"], 2
)
if tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]:
    tracking_data["left_arm"]["rep_metrics"]["avg_shoulder_stability"] = round(
        np.mean(tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )
    tracking_data["left_arm"]["rep_metrics"]["max_shoulder_movement"] = round(
        max(tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )

# Find the peak curl point for left arm
left_peak_curl_frame = min(tracking_data["left_arm"]["frames"], key=lambda x: x["elbow_angle"])
tracking_data["left_arm"]["rep_metrics"]["peak_curl"] = left_peak_curl_frame

# RIGHT ARM METRICS
tracking_data["right_arm"]["rep_metrics"]["angle_range"] = round(
    tracking_data["right_arm"]["rep_metrics"]["max_elbow_angle"] - 
    tracking_data["right_arm"]["rep_metrics"]["min_elbow_angle"], 2
)
if tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]:
    tracking_data["right_arm"]["rep_metrics"]["avg_shoulder_stability"] = round(
        np.mean(tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )
    tracking_data["right_arm"]["rep_metrics"]["max_shoulder_movement"] = round(
        max(tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )

# Find the peak curl point for right arm
right_peak_curl_frame = min(tracking_data["right_arm"]["frames"], key=lambda x: x["elbow_angle"])
tracking_data["right_arm"]["rep_metrics"]["peak_curl"] = right_peak_curl_frame

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("=== ANALYSIS COMPLETE ===")
print("="*80)
print(f"\nTotal Frames Analyzed: {processed_frames}")

print("\n" + "="*80)
print("*** LEFT ARM - EXTRACTED METRICS ***")
print("="*80)
print(f"  • Starting Elbow Angle (DOWN position): {tracking_data['left_arm']['frames'][0]['elbow_angle']}°")
print(f"  • Peak Curl Angle (UP position): {tracking_data['left_arm']['rep_metrics']['min_elbow_angle']}°")
print(f"  • Ending Elbow Angle: {tracking_data['left_arm']['frames'][-1]['elbow_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['left_arm']['rep_metrics']['angle_range']}°")
print(f"  • Average Shoulder Stability: {tracking_data['left_arm']['rep_metrics']['avg_shoulder_stability']}")
print(f"  • Max Shoulder Movement: {tracking_data['left_arm']['rep_metrics']['max_shoulder_movement']}")
print(f"  • Peak Curl at frame {left_peak_curl_frame['frame_number']} ({left_peak_curl_frame['time_seconds']}s)")

print("\n" + "="*80)
print("*** RIGHT ARM - EXTRACTED METRICS ***")
print("="*80)
print(f"  • Starting Elbow Angle (DOWN position): {tracking_data['right_arm']['frames'][0]['elbow_angle']}°")
print(f"  • Peak Curl Angle (UP position): {tracking_data['right_arm']['rep_metrics']['min_elbow_angle']}°")
print(f"  • Ending Elbow Angle: {tracking_data['right_arm']['frames'][-1]['elbow_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['right_arm']['rep_metrics']['angle_range']}°")
print(f"  • Average Shoulder Stability: {tracking_data['right_arm']['rep_metrics']['avg_shoulder_stability']}")
print(f"  • Max Shoulder Movement: {tracking_data['right_arm']['rep_metrics']['max_shoulder_movement']}")
print(f"  • Peak Curl at frame {right_peak_curl_frame['frame_number']} ({right_peak_curl_frame['time_seconds']}s)")

# ============================================================================
# Save detailed metrics to text file
# ============================================================================
with open('bicep_curl_metrics_both_arms.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BICEP CURL METRICS - BOTH ARMS - USE THESE IN YOUR COUNTER\n")
    f.write("="*80 + "\n\n")
    
    f.write("VIDEO INFO:\n")
    f.write(f"  Filename: {tracking_data['video_info']['filename']}\n")
    f.write(f"  Frames Analyzed: {tracking_data['video_info']['total_frames_analyzed']}\n\n")
    
    f.write("="*80 + "\n")
    f.write("LEFT ARM - THRESHOLDS FOR COUNTER\n")
    f.write("="*80 + "\n")
    f.write(f"  DOWN_THRESHOLD (arm extended): {tracking_data['left_arm']['frames'][0]['elbow_angle']}°\n")
    f.write(f"  UP_THRESHOLD (peak curl): {tracking_data['left_arm']['rep_metrics']['min_elbow_angle']}°\n")
    f.write(f"  MIN_RANGE_OF_MOTION: {tracking_data['left_arm']['rep_metrics']['angle_range']}°\n")
    f.write(f"  MAX_SHOULDER_MOVEMENT: {tracking_data['left_arm']['rep_metrics']['max_shoulder_movement']}\n\n")
    
    f.write("LEFT ARM - INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['left_arm']['initial_position']['shoulder']['x']}, y={tracking_data['left_arm']['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['left_arm']['initial_position']['elbow']['x']}, y={tracking_data['left_arm']['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['left_arm']['initial_position']['wrist']['x']}, y={tracking_data['left_arm']['initial_position']['wrist']['y']}\n")
    f.write(f"  Elbow Angle: {tracking_data['left_arm']['initial_position']['elbow_angle']}°\n\n")
    
    f.write("="*80 + "\n")
    f.write("RIGHT ARM - THRESHOLDS FOR COUNTER\n")
    f.write("="*80 + "\n")
    f.write(f"  DOWN_THRESHOLD (arm extended): {tracking_data['right_arm']['frames'][0]['elbow_angle']}°\n")
    f.write(f"  UP_THRESHOLD (peak curl): {tracking_data['right_arm']['rep_metrics']['min_elbow_angle']}°\n")
    f.write(f"  MIN_RANGE_OF_MOTION: {tracking_data['right_arm']['rep_metrics']['angle_range']}°\n")
    f.write(f"  MAX_SHOULDER_MOVEMENT: {tracking_data['right_arm']['rep_metrics']['max_shoulder_movement']}\n\n")
    
    f.write("RIGHT ARM - INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['right_arm']['initial_position']['shoulder']['x']}, y={tracking_data['right_arm']['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['right_arm']['initial_position']['elbow']['x']}, y={tracking_data['right_arm']['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['right_arm']['initial_position']['wrist']['x']}, y={tracking_data['right_arm']['initial_position']['wrist']['y']}\n")
    f.write(f"  Elbow Angle: {tracking_data['right_arm']['initial_position']['elbow_angle']}°\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("LEFT ARM - FRAME-BY-FRAME DATA\n")
    f.write("="*80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Elbow°':<10} {'Shoulder°':<12} {'Shoulder Drift':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['left_arm']['frames']):
        shoulder_drift = tracking_data['left_arm']['rep_metrics']['shoulder_stability'][i] if i < len(tracking_data['left_arm']['rep_metrics']['shoulder_stability']) else 0
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['elbow_angle']:<10} "
                f"{frame['shoulder_angle']:<12} {shoulder_drift:<15}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RIGHT ARM - FRAME-BY-FRAME DATA\n")
    f.write("="*80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Elbow°':<10} {'Shoulder°':<12} {'Shoulder Drift':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['right_arm']['frames']):
        shoulder_drift = tracking_data['right_arm']['rep_metrics']['shoulder_stability'][i] if i < len(tracking_data['right_arm']['rep_metrics']['shoulder_stability']) else 0
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['elbow_angle']:<10} "
                f"{frame['shoulder_angle']:<12} {shoulder_drift:<15}\n")

# Save JSON for programmatic use
with open('bicep_curl_metrics_both_arms.json', 'w') as f:
    json.dump(tracking_data, f, indent=2)

print("\n" + "="*80)
print("✓ Metrics saved to 'bicep_curl_metrics_both_arms.txt'")
print("✓ Detailed JSON saved to 'bicep_curl_metrics_both_arms.json'")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Check 'bicep_curl_metrics_both_arms.txt' for the extracted thresholds")
print("2. Use these values in your bicep curl counter script")
print("3. You now have metrics for BOTH left and right arms!")
print("="*80)