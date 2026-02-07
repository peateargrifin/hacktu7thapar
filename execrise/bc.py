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

# Initialize MediaPipe
mp_pose = mp.solutions.pose

# Load video
video_path = "vids/Denise_Wide-Grip20Curl.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate frame interval - ensure it's at least 1
if fps >= 15:
    frame_interval = int(fps / 15)  # Process at 15 FPS
else:
    frame_interval = 1  # Process every frame if video is slower than 15 FPS
    
actual_sampling_fps = fps / frame_interval if frame_interval > 0 else fps

print(f"Video FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Frame Interval: {frame_interval}")
print(f"Actual Sampling Rate: ~{actual_sampling_fps:.2f} FPS")

# Storage for tracking data
tracking_data = {
    "video_info": {
        "filename": "Denise_Wide-Grip20Curl.mp4",
        "original_fps": fps,
        "sampling_fps": actual_sampling_fps,
        "total_frames_analyzed": 0,
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
                # Get landmark positions (using right side - adjust if needed)
                shoulder = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                elbow = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
                wrist = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
                hip = get_landmark_coords(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                
                # Calculate elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Calculate shoulder angle (shoulder-hip-vertical)
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

print("\n" + "="*60)
print("=== ANALYSIS COMPLETE ===")
print("="*60)
print(f"\nTotal Frames Analyzed: {processed_frames}")
print(f"\nREP METRICS (Perfect Form Reference):")
print(f"  • Starting Elbow Angle: {tracking_data['frames'][0]['elbow_angle']}°")
print(f"  • Peak Curl Angle: {tracking_data['rep_metrics']['min_elbow_angle']}°")
print(f"  • Ending Elbow Angle: {tracking_data['frames'][-1]['elbow_angle']}°")
print(f"  • Full Range of Motion: {tracking_data['rep_metrics']['angle_range']}°")
print(f"  • Average Shoulder Stability: {tracking_data['rep_metrics']['avg_shoulder_stability']}")
print(f"  • Max Shoulder Movement: {tracking_data['rep_metrics']['max_shoulder_movement']}")
print(f"\nPeak Curl occurred at frame {peak_curl_frame['frame_number']} ({peak_curl_frame['time_seconds']}s)")

# Append to positions.txt
with open('positions.txt', 'a') as f:
    f.write("\n" + "="*80 + "\n")
    f.write(f"PERFECT FORM REFERENCE - {tracking_data['video_info']['timestamp']}\n")
    f.write("="*80 + "\n\n")
    
    f.write("VIDEO INFO:\n")
    f.write(f"  Filename: {tracking_data['video_info']['filename']}\n")
    f.write(f"  Original FPS: {tracking_data['video_info']['original_fps']}\n")
    f.write(f"  Sampling Rate: {tracking_data['video_info']['sampling_fps']} FPS\n")
    f.write(f"  Frames Analyzed: {tracking_data['video_info']['total_frames_analyzed']}\n\n")
    
    f.write("INITIAL POSITION:\n")
    f.write(f"  Shoulder: x={tracking_data['initial_position']['shoulder']['x']}, y={tracking_data['initial_position']['shoulder']['y']}\n")
    f.write(f"  Elbow: x={tracking_data['initial_position']['elbow']['x']}, y={tracking_data['initial_position']['elbow']['y']}\n")
    f.write(f"  Wrist: x={tracking_data['initial_position']['wrist']['x']}, y={tracking_data['initial_position']['wrist']['y']}\n")
    f.write(f"  Elbow Angle: {tracking_data['initial_position']['elbow_angle']}°\n")
    f.write(f"  Shoulder Angle: {tracking_data['initial_position']['shoulder_angle']}°\n\n")
    
    f.write("REP METRICS (PERFECT FORM THRESHOLDS):\n")
    f.write(f"  Starting Elbow Angle: {tracking_data['frames'][0]['elbow_angle']}°\n")
    f.write(f"  Peak Curl Angle (Minimum): {tracking_data['rep_metrics']['min_elbow_angle']}°\n")
    f.write(f"  Ending Elbow Angle: {tracking_data['frames'][-1]['elbow_angle']}°\n")
    f.write(f"  Full Range of Motion: {tracking_data['rep_metrics']['angle_range']}°\n")
    f.write(f"  Average Shoulder Stability: {tracking_data['rep_metrics']['avg_shoulder_stability']}\n")
    f.write(f"  Max Shoulder Movement Allowed: {tracking_data['rep_metrics']['max_shoulder_movement']}\n\n")
    
    f.write("FRAME-BY-FRAME TRACKING (15 FPS):\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Frame':<8} {'Time(s)':<10} {'Elbow°':<10} {'Shoulder°':<12} {'Shoulder Drift':<15}\n")
    f.write("-" * 80 + "\n")
    
    for i, frame in enumerate(tracking_data['frames']):
        shoulder_drift = tracking_data['rep_metrics']['shoulder_stability'][i]
        f.write(f"{frame['frame_number']:<8} {frame['time_seconds']:<10} {frame['elbow_angle']:<10} "
                f"{frame['shoulder_angle']:<12} {shoulder_drift:<15}\n")
    
    f.write("\n" + "="*80 + "\n\n")

# Also save detailed JSON for programmatic use
with open('perfect_form_data.json', 'w') as f:
    json.dump(tracking_data, f, indent=2)

print(f"\n✓ Data appended to 'positions.txt'")
print(f"✓ Detailed JSON saved to 'perfect_form_data.json'")
print("\nYou can now use these metrics in your exercise tracker to validate bicep curl form!")