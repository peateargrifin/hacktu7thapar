"""
STEP 1: Side Lateral Raises Video Analysis - BOTH ARMS SIMULTANEOUSLY
Run this script on your perfect form lateral raises video to extract metrics for BOTH arms
Arms move TOGETHER: UP (raise to sides) then DOWN (return to sides)
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
VIDEO_PATH = "vids/sl.mp4"  # <-- CHANGE THIS TO YOUR VIDEO
# ============================================================================

# Initialize MediaPipe
mp_pose = mp.solutions.pose

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video file: {VIDEO_PATH}")
    print("Please update VIDEO_PATH in the script to point to your lateral raises video")
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
print("SIDE LATERAL RAISES VIDEO ANALYSIS - BOTH ARMS SIMULTANEOUSLY")
print("="*80)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Original FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Frame Interval: {frame_interval}")
print(f"Sampling Rate: ~{actual_sampling_fps:.2f} FPS")
print(f"Analyzing: BOTH arms moving together (UP then DOWN)")
print("="*80)

# Storage for tracking data - both arms analyzed together
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
            "min_shoulder_angle": 180,  # Lowest angle (arms down)
            "max_shoulder_angle": 0,     # Highest angle (arms up)
            "angle_range": 0,
            "avg_elbow_angle": 0,
            "elbow_variance": [],
            "shoulder_stability": [],    # Track shoulder elevation (shrugging)
            "elbow_angles": []
        }
    },
    "right_arm": {
        "frames": [],
        "initial_position": {},
        "rep_metrics": {
            "min_shoulder_angle": 180,
            "max_shoulder_angle": 0,
            "angle_range": 0,
            "avg_elbow_angle": 0,
            "elbow_variance": [],
            "shoulder_stability": [],
            "elbow_angles": []
        }
    },
    "symmetry_metrics": {
        "angle_differences": [],  # Left vs Right shoulder angle differences
        "max_asymmetry": 0,
        "avg_asymmetry": 0
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
                # Shoulder angle: hip-shoulder-elbow (measures how high arm is raised)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                # Elbow angle: shoulder-elbow-wrist (should stay relatively constant ~160-175°)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Store left arm frame data
                left_frame_data = {
                    "frame_number": frame_count,
                    "time_seconds": round(frame_count / fps, 2),
                    "shoulder": {"x": round(left_shoulder[0], 4), "y": round(left_shoulder[1], 4)},
                    "elbow": {"x": round(left_elbow[0], 4), "y": round(left_elbow[1], 4)},
                    "wrist": {"x": round(left_wrist[0], 4), "y": round(left_wrist[1], 4)},
                    "hip": {"x": round(left_hip[0], 4), "y": round(left_hip[1], 4)},
                    "shoulder_angle": round(left_shoulder_angle, 2),  # Main metric for raise height
                    "elbow_angle": round(left_elbow_angle, 2)         # Should stay consistent
                }
                
                tracking_data["left_arm"]["frames"].append(left_frame_data)
                tracking_data["left_arm"]["rep_metrics"]["elbow_angles"].append(left_elbow_angle)
                
                # Track min/max shoulder angles (down/up positions)
                if left_shoulder_angle < tracking_data["left_arm"]["rep_metrics"]["min_shoulder_angle"]:
                    tracking_data["left_arm"]["rep_metrics"]["min_shoulder_angle"] = round(left_shoulder_angle, 2)
                if left_shoulder_angle > tracking_data["left_arm"]["rep_metrics"]["max_shoulder_angle"]:
                    tracking_data["left_arm"]["rep_metrics"]["max_shoulder_angle"] = round(left_shoulder_angle, 2)
                
                # Store initial position for left arm (first frame - arms down)
                if processed_frames == 0:
                    tracking_data["left_arm"]["initial_position"] = left_frame_data.copy()
                    print(f"\n=== LEFT ARM - INITIAL POSITION (Arms Down - Frame {frame_count}) ===")
                    print(f"Shoulder: {left_shoulder}")
                    print(f"Elbow: {left_elbow}")
                    print(f"Wrist: {left_wrist}")
                    print(f"Shoulder Angle (raise height): {left_shoulder_angle:.2f}°")
                    print(f"Elbow Angle (should stay ~160-175°): {left_elbow_angle:.2f}°")
                
                # Track left shoulder stability (vertical movement - shrugging detection)
                if processed_frames > 0:
                    left_shoulder_elevation = abs(
                        left_shoulder[1] - tracking_data["left_arm"]["initial_position"]["shoulder"]["y"]
                    )
                    tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"].append(round(left_shoulder_elevation, 4))
                
                # ============================================================
                # PROCESS RIGHT ARM
                # ============================================================
                right_shoulder = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                right_elbow = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
                right_wrist = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
                right_hip = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                
                # Calculate right arm angles
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Store right arm frame data
                right_frame_data = {
                    "frame_number": frame_count,
                    "time_seconds": round(frame_count / fps, 2),
                    "shoulder": {"x": round(right_shoulder[0], 4), "y": round(right_shoulder[1], 4)},
                    "elbow": {"x": round(right_elbow[0], 4), "y": round(right_elbow[1], 4)},
                    "wrist": {"x": round(right_wrist[0], 4), "y": round(right_wrist[1], 4)},
                    "hip": {"x": round(right_hip[0], 4), "y": round(right_hip[1], 4)},
                    "shoulder_angle": round(right_shoulder_angle, 2),
                    "elbow_angle": round(right_elbow_angle, 2)
                }
                
                tracking_data["right_arm"]["frames"].append(right_frame_data)
                tracking_data["right_arm"]["rep_metrics"]["elbow_angles"].append(right_elbow_angle)
                
                # Track min/max shoulder angles
                if right_shoulder_angle < tracking_data["right_arm"]["rep_metrics"]["min_shoulder_angle"]:
                    tracking_data["right_arm"]["rep_metrics"]["min_shoulder_angle"] = round(right_shoulder_angle, 2)
                if right_shoulder_angle > tracking_data["right_arm"]["rep_metrics"]["max_shoulder_angle"]:
                    tracking_data["right_arm"]["rep_metrics"]["max_shoulder_angle"] = round(right_shoulder_angle, 2)
                
                # Store initial position for right arm
                if processed_frames == 0:
                    tracking_data["right_arm"]["initial_position"] = right_frame_data.copy()
                    print(f"\n=== RIGHT ARM - INITIAL POSITION (Arms Down - Frame {frame_count}) ===")
                    print(f"Shoulder: {right_shoulder}")
                    print(f"Elbow: {right_elbow}")
                    print(f"Wrist: {right_wrist}")
                    print(f"Shoulder Angle (raise height): {right_shoulder_angle:.2f}°")
                    print(f"Elbow Angle (should stay ~160-175°): {right_elbow_angle:.2f}°")
                
                # Track right shoulder stability
                if processed_frames > 0:
                    right_shoulder_elevation = abs(
                        right_shoulder[1] - tracking_data["right_arm"]["initial_position"]["shoulder"]["y"]
                    )
                    tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"].append(round(right_shoulder_elevation, 4))
                
                # ============================================================
                # CALCULATE SYMMETRY (Left vs Right comparison)
                # ============================================================
                angle_diff = abs(left_shoulder_angle - right_shoulder_angle)
                tracking_data["symmetry_metrics"]["angle_differences"].append(round(angle_diff, 2))
                
                processed_frames += 1
                
                if processed_frames % 15 == 0:  # Print every second
                    print(f"Processed {processed_frames} frames ({frame_count}/{total_frames}) - "
                          f"Left: {left_shoulder_angle:.2f}° | Right: {right_shoulder_angle:.2f}° | "
                          f"Diff: {angle_diff:.2f}°")

        frame_count += 1

cap.release()

# ============================================================================
# Calculate final metrics for BOTH arms
# ============================================================================
tracking_data["video_info"]["total_frames_analyzed"] = processed_frames

# LEFT ARM METRICS
tracking_data["left_arm"]["rep_metrics"]["angle_range"] = round(
    tracking_data["left_arm"]["rep_metrics"]["max_shoulder_angle"] - 
    tracking_data["left_arm"]["rep_metrics"]["min_shoulder_angle"], 2
)

if tracking_data["left_arm"]["rep_metrics"]["elbow_angles"]:
    tracking_data["left_arm"]["rep_metrics"]["avg_elbow_angle"] = round(
        np.mean(tracking_data["left_arm"]["rep_metrics"]["elbow_angles"]), 2
    )
    tracking_data["left_arm"]["rep_metrics"]["elbow_std_dev"] = round(
        np.std(tracking_data["left_arm"]["rep_metrics"]["elbow_angles"]), 2
    )

if tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]:
    tracking_data["left_arm"]["rep_metrics"]["avg_shoulder_elevation"] = round(
        np.mean(tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )
    tracking_data["left_arm"]["rep_metrics"]["max_shoulder_elevation"] = round(
        max(tracking_data["left_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )

# Find the peak raise point for left arm (max shoulder angle)
left_peak_raise_frame = max(tracking_data["left_arm"]["frames"], key=lambda x: x["shoulder_angle"])
tracking_data["left_arm"]["rep_metrics"]["peak_raise"] = left_peak_raise_frame

# RIGHT ARM METRICS
tracking_data["right_arm"]["rep_metrics"]["angle_range"] = round(
    tracking_data["right_arm"]["rep_metrics"]["max_shoulder_angle"] - 
    tracking_data["right_arm"]["rep_metrics"]["min_shoulder_angle"], 2
)

if tracking_data["right_arm"]["rep_metrics"]["elbow_angles"]:
    tracking_data["right_arm"]["rep_metrics"]["avg_elbow_angle"] = round(
        np.mean(tracking_data["right_arm"]["rep_metrics"]["elbow_angles"]), 2
    )
    tracking_data["right_arm"]["rep_metrics"]["elbow_std_dev"] = round(
        np.std(tracking_data["right_arm"]["rep_metrics"]["elbow_angles"]), 2
    )

if tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]:
    tracking_data["right_arm"]["rep_metrics"]["avg_shoulder_elevation"] = round(
        np.mean(tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )
    tracking_data["right_arm"]["rep_metrics"]["max_shoulder_elevation"] = round(
        max(tracking_data["right_arm"]["rep_metrics"]["shoulder_stability"]), 4
    )

# Find the peak raise point for right arm
right_peak_raise_frame = max(tracking_data["right_arm"]["frames"], key=lambda x: x["shoulder_angle"])
tracking_data["right_arm"]["rep_metrics"]["peak_raise"] = right_peak_raise_frame

# SYMMETRY METRICS
if tracking_data["symmetry_metrics"]["angle_differences"]:
    tracking_data["symmetry_metrics"]["avg_asymmetry"] = round(
        np.mean(tracking_data["symmetry_metrics"]["angle_differences"]), 2
    )
    tracking_data["symmetry_metrics"]["max_asymmetry"] = round(
        max(tracking_data["symmetry_metrics"]["angle_differences"]), 2
    )

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
print(f"  • Starting Shoulder Angle (Arms DOWN): {tracking_data['left_arm']['frames'][0]['shoulder_angle']}°")
print(f"  • Peak Shoulder Angle (Arms UP/Raised): {tracking_data['left_arm']['rep_metrics']['max_shoulder_angle']}°")
print(f"  • Ending Shoulder Angle: {tracking_data['left_arm']['frames'][-1]['shoulder_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['left_arm']['rep_metrics']['angle_range']}°")
print(f"  • Average Elbow Angle (consistency): {tracking_data['left_arm']['rep_metrics']['avg_elbow_angle']}°")
print(f"  • Elbow Variance (std dev): {tracking_data['left_arm']['rep_metrics']['elbow_std_dev']}°")
print(f"  • Average Shoulder Elevation: {tracking_data['left_arm']['rep_metrics']['avg_shoulder_elevation']}")
print(f"  • Max Shoulder Elevation (shrugging): {tracking_data['left_arm']['rep_metrics']['max_shoulder_elevation']}")
print(f"  • Peak Raise at frame {left_peak_raise_frame['frame_number']} ({left_peak_raise_frame['time_seconds']}s)")

print("\n" + "="*80)
print("*** RIGHT ARM - EXTRACTED METRICS ***")
print("="*80)
print(f"  • Starting Shoulder Angle (Arms DOWN): {tracking_data['right_arm']['frames'][0]['shoulder_angle']}°")
print(f"  • Peak Shoulder Angle (Arms UP/Raised): {tracking_data['right_arm']['rep_metrics']['max_shoulder_angle']}°")
print(f"  • Ending Shoulder Angle: {tracking_data['right_arm']['frames'][-1]['shoulder_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['right_arm']['rep_metrics']['angle_range']}°")
print(f"  • Average Elbow Angle (consistency): {tracking_data['right_arm']['rep_metrics']['avg_elbow_angle']}°")
print(f"  • Elbow Variance (std dev): {tracking_data['right_arm']['rep_metrics']['elbow_std_dev']}°")
print(f"  • Average Shoulder Elevation: {tracking_data['right_arm']['rep_metrics']['avg_shoulder_elevation']}")
print(f"  • Max Shoulder Elevation (shrugging): {tracking_data['right_arm']['rep_metrics']['max_shoulder_elevation']}")
print(f"  • Peak Raise at frame {right_peak_raise_frame['frame_number']} ({right_peak_raise_frame['time_seconds']}s)")

print("\n" + "="*80)
print("*** SYMMETRY ANALYSIS (Left vs Right) ***")
print("="*80)
print(f"  • Average Asymmetry: {tracking_data['symmetry_metrics']['avg_asymmetry']}°")
print(f"  • Max Asymmetry: {tracking_data['symmetry_metrics']['max_asymmetry']}°")
print(f"  • Symmetry Quality: {'EXCELLENT' if tracking_data['symmetry_metrics']['avg_asymmetry'] < 5 else 'GOOD' if tracking_data['symmetry_metrics']['avg_asymmetry'] < 10 else 'NEEDS WORK'}")

# ============================================================================
# Save detailed metrics to text file
# ============================================================================
with open('lateral_raises_metrics_both_arms.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SIDE LATERAL RAISES METRICS - BOTH ARMS - USE THESE IN YOUR COUNTER\n")
    f.write("="*80 + "\n\n")
    
    f.write("VIDEO INFO:\n")
    f.write(f"  Filename: {tracking_data['video_info']['filename']}\n")
    f.write(f"  Frames Analyzed: {tracking_data['video_info']['total_frames_analyzed']}\n\n")
    
    f.write("="*80 + "\n")
    f.write("LEFT ARM - THRESHOLDS FOR COUNTER\n")
    f.write("="*80 + "\n")
    f.write(f"  DOWN_THRESHOLD (arms at sides): {tracking_data['left_arm']['frames'][0]['shoulder_angle']}°\n")
    f.write(f"  UP_THRESHOLD (arms raised): {tracking_data['left_arm']['rep_metrics']['max_shoulder_angle']}°\n")
    f.write(f"  MIN_RANGE_OF_MOTION: {tracking_data['left_arm']['rep_metrics']['angle_range']}°\n")
    f.write(f"  TARGET_ELBOW_ANGLE: {tracking_data['left_arm']['rep_metrics']['avg_elbow_angle']}°\n")
    f.write(f"  MAX_ELBOW_VARIANCE: {tracking_data['left_arm']['rep_metrics']['elbow_std_dev']}°\n")
    f.write(f"  MAX_SHOULDER_ELEVATION: {tracking_data['left_arm']['rep_metrics']['max_shoulder_elevation']}\n\n")
    
    f.write("LEFT ARM - INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['left_arm']['initial_position']['shoulder']['x']}, y={tracking_data['left_arm']['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['left_arm']['initial_position']['elbow']['x']}, y={tracking_data['left_arm']['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['left_arm']['initial_position']['wrist']['x']}, y={tracking_data['left_arm']['initial_position']['wrist']['y']}\n")
    f.write(f"  Shoulder Angle: {tracking_data['left_arm']['initial_position']['shoulder_angle']}°\n")
    f.write(f"  Elbow Angle: {tracking_data['left_arm']['initial_position']['elbow_angle']}°\n\n")
    
    f.write("="*80 + "\n")
    f.write("RIGHT ARM - THRESHOLDS FOR COUNTER\n")
    f.write("="*80 + "\n")
    f.write(f"  DOWN_THRESHOLD (arms at sides): {tracking_data['right_arm']['frames'][0]['shoulder_angle']}°\n")
    f.write(f"  UP_THRESHOLD (arms raised): {tracking_data['right_arm']['rep_metrics']['max_shoulder_angle']}°\n")
    f.write(f"  MIN_RANGE_OF_MOTION: {tracking_data['right_arm']['rep_metrics']['angle_range']}°\n")
    f.write(f"  TARGET_ELBOW_ANGLE: {tracking_data['right_arm']['rep_metrics']['avg_elbow_angle']}°\n")
    f.write(f"  MAX_ELBOW_VARIANCE: {tracking_data['right_arm']['rep_metrics']['elbow_std_dev']}°\n")
    f.write(f"  MAX_SHOULDER_ELEVATION: {tracking_data['right_arm']['rep_metrics']['max_shoulder_elevation']}\n\n")
    
    f.write("RIGHT ARM - INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['right_arm']['initial_position']['shoulder']['x']}, y={tracking_data['right_arm']['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['right_arm']['initial_position']['elbow']['x']}, y={tracking_data['right_arm']['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['right_arm']['initial_position']['wrist']['x']}, y={tracking_data['right_arm']['initial_position']['wrist']['y']}\n")
    f.write(f"  Shoulder Angle: {tracking_data['right_arm']['initial_position']['shoulder_angle']}°\n")
    f.write(f"  Elbow Angle: {tracking_data['right_arm']['initial_position']['elbow_angle']}°\n\n")
    
    f.write("="*80 + "\n")
    f.write("SYMMETRY THRESHOLDS\n")
    f.write("="*80 + "\n")
    f.write(f"  EXCELLENT_SYMMETRY: < 5° difference\n")
    f.write(f"  GOOD_SYMMETRY: < 10° difference\n")
    f.write(f"  Average Asymmetry from video: {tracking_data['symmetry_metrics']['avg_asymmetry']}°\n")
    f.write(f"  Max Asymmetry from video: {tracking_data['symmetry_metrics']['max_asymmetry']}°\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("LEFT ARM - FRAME-BY-FRAME DATA\n")
    f.write("="*80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Shoulder°':<12} {'Elbow°':<10} {'Shoulder Elev':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['left_arm']['frames']):
        shoulder_elev = tracking_data['left_arm']['rep_metrics']['shoulder_stability'][i] if i < len(tracking_data['left_arm']['rep_metrics']['shoulder_stability']) else 0
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['shoulder_angle']:<12} "
                f"{frame['elbow_angle']:<10} {shoulder_elev:<15}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RIGHT ARM - FRAME-BY-FRAME DATA\n")
    f.write("="*80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Shoulder°':<12} {'Elbow°':<10} {'Shoulder Elev':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['right_arm']['frames']):
        shoulder_elev = tracking_data['right_arm']['rep_metrics']['shoulder_stability'][i] if i < len(tracking_data['right_arm']['rep_metrics']['shoulder_stability']) else 0
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['shoulder_angle']:<12} "
                f"{frame['elbow_angle']:<10} {shoulder_elev:<15}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("SYMMETRY - FRAME-BY-FRAME\n")
    f.write("="*80 + "\n")
    f.write(f"{'Frame':<8} {'Left°':<10} {'Right°':<10} {'Difference°':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i in range(min(len(tracking_data['left_arm']['frames']), len(tracking_data['right_arm']['frames']))):
        left_angle = tracking_data['left_arm']['frames'][i]['shoulder_angle']
        right_angle = tracking_data['right_arm']['frames'][i]['shoulder_angle']
        diff = tracking_data['symmetry_metrics']['angle_differences'][i] if i < len(tracking_data['symmetry_metrics']['angle_differences']) else 0
        f.write(f"{tracking_data['left_arm']['frames'][i]['frame_number']:<8} {left_angle:<10} "
                f"{right_angle:<10} {diff:<15}\n")

# Save JSON for programmatic use
with open('lateral_raises_metrics_both_arms.json', 'w') as f:
    json.dump(tracking_data, f, indent=2)

print("\n" + "="*80)
print("✓ Metrics saved to 'lateral_raises_metrics_both_arms.txt'")
print("✓ Detailed JSON saved to 'lateral_raises_metrics_both_arms.json'")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Check 'lateral_raises_metrics_both_arms.txt' for the extracted thresholds")
print("2. Send me the numbers and I'll create your lateral raises counter script")
print("3. The counter will track BOTH arms simultaneously and detect:")
print("   - Proper raise height (shoulder angle)")
print("   - Elbow consistency (should stay slightly bent)")
print("   - Left-Right symmetry")
print("   - Shoulder shrugging (form issue)")
print("="*80)