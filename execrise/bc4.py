"""
BICEP CURL COUNTER WITH FORM ANALYSIS
Based on analyzed perfect form metrics from video analysis

Step 1: Metrics extracted from perfect form video analysis
Step 2: Implementation of counter with these metrics
"""

import cv2
import mediapipe as mp
import numpy as np

# ============================================================================
# STEP 1: METRICS FROM PERFECT FORM VIDEO ANALYSIS
# ============================================================================
# These would come from running the analysis script on a perfect form video
# Example metrics from a typical bicep curl analysis:

PERFECT_FORM_METRICS = {
    "starting_elbow_angle": 165,      # Almost fully extended
    "peak_curl_angle": 35,             # Fully curled position
    "full_range_of_motion": 130,      # Difference between start and peak
    "avg_shoulder_stability": 0.015,  # Very minimal shoulder movement
    "max_shoulder_movement": 0.035,   # Maximum allowed drift
}

# Derived thresholds for real-time counter
CURL_THRESHOLDS = {
    "down_position": 160,              # Arm is down when angle > 160°
    "up_position": 40,                 # Arm is up when angle < 40°
    "min_rom_for_valid_rep": 100,     # Minimum range needed for a good rep
    "excellent_peak": 40,              # Excellent curl depth
    "good_peak": 55,                   # Good curl depth
    "max_shoulder_drift": 0.04,        # Shoulder stability threshold
}


# ============================================================================
# STEP 2: HELPER FUNCTIONS
# ============================================================================

def calculate_angle(a, b, c):
    """
    Calculate angle between three points (a-b-c where b is the vertex)
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y] (vertex)
        c: End point [x, y]
    
    Returns:
        angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def get_landmark_coords(landmarks, landmark_id):
    """Extract x, y coordinates from a landmark"""
    landmark = landmarks[landmark_id]
    return [landmark.x, landmark.y]


def calculate_shoulder_drift(current_shoulder, initial_shoulder):
    """Calculate how much the shoulder has moved from initial position"""
    return np.sqrt(
        (current_shoulder[0] - initial_shoulder[0])**2 +
        (current_shoulder[1] - initial_shoulder[1])**2
    )


# ============================================================================
# STEP 3: MAIN BICEP CURL COUNTER
# ============================================================================

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

# Counter variables
counter = 0
stage = None

# Form tracking variables
initial_shoulder = None
form_warning = ""  # FIX for your error!

# Rep quality tracking
good_reps = 0
partial_reps = 0

# Current rep metrics
current_rep = {
    "min_angle": 180,
    "max_angle": 0,
    "shoulder_drift": 0,
}

print("="*60)
print("BICEP CURL COUNTER - Based on Perfect Form Analysis")
print("="*60)
print(f"Target Metrics:")
print(f"  • Down Position: >{CURL_THRESHOLDS['down_position']}°")
print(f"  • Peak Curl: <{CURL_THRESHOLDS['up_position']}°")
print(f"  • Minimum ROM: {CURL_THRESHOLDS['min_rom_for_valid_rep']}°")
print(f"  • Shoulder Stability: <{CURL_THRESHOLDS['max_shoulder_drift']:.3f}")
print("="*60)
print("\nPress 'Q' to quit\n")

# Setup MediaPipe Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and process
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates (LEFT arm - change to RIGHT if needed)
            shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
            wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)

            # Set initial shoulder position (first frame)
            if initial_shoulder is None:
                initial_shoulder = shoulder.copy()
                print("Initial position captured!")

            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Track min/max for current rep
            if elbow_angle < current_rep["min_angle"]:
                current_rep["min_angle"] = elbow_angle
            if elbow_angle > current_rep["max_angle"]:
                current_rep["max_angle"] = elbow_angle

            # Calculate shoulder drift
            shoulder_drift = calculate_shoulder_drift(shoulder, initial_shoulder)
            current_rep["shoulder_drift"] = max(current_rep["shoulder_drift"], shoulder_drift)

            # Reset form warning
            form_warning = ""

            # Check shoulder stability
            if shoulder_drift > CURL_THRESHOLDS["max_shoulder_drift"]:
                form_warning = "Keep shoulders stable!"

            # Display angle on elbow
            cv2.putText(
                image, 
                f'{int(elbow_angle)}°',
                tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

            # BICEP CURL COUNTER LOGIC
            if elbow_angle > CURL_THRESHOLDS["down_position"]:
                stage = "down"
                
            if elbow_angle < CURL_THRESHOLDS["up_position"] and stage == 'down':
                # Rep completed - evaluate form
                rom = current_rep["max_angle"] - current_rep["min_angle"]
                peak_angle = current_rep["min_angle"]
                shoulder_stable = current_rep["shoulder_drift"] < CURL_THRESHOLDS["max_shoulder_drift"]
                
                # Determine rep quality
                if rom >= CURL_THRESHOLDS["min_rom_for_valid_rep"] and shoulder_stable:
                    if peak_angle <= CURL_THRESHOLDS["excellent_peak"]:
                        form_quality = "EXCELLENT"
                        good_reps += 1
                    elif peak_angle <= CURL_THRESHOLDS["good_peak"]:
                        form_quality = "GOOD"
                        good_reps += 1
                    else:
                        form_quality = "PARTIAL"
                        partial_reps += 1
                else:
                    form_quality = "INCOMPLETE"
                    partial_reps += 1
                    if not shoulder_stable:
                        form_quality += " (SHOULDERS)"
                    if rom < CURL_THRESHOLDS["min_rom_for_valid_rep"]:
                        form_quality += " (ROM)"
                
                stage = "up"
                counter += 1
                
                # Print rep summary
                print(f"Rep #{counter}: {form_quality} | "
                      f"ROM={rom:.1f}° | Peak={peak_angle:.1f}° | "
                      f"Shoulder Drift={current_rep['shoulder_drift']:.3f}")
                
                # Reset for next rep
                current_rep = {
                    "min_angle": 180,
                    "max_angle": 0,
                    "shoulder_drift": 0,
                }

        except Exception as e:
            pass

        # ====================================================================
        # RENDER UI ELEMENTS
        # ====================================================================
        
        # Status box background
        cv2.rectangle(image, (0, 0), (400, 200), (245, 117, 16), -1)

        # Rep counter
        cv2.putText(image, 'REPS', (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        # Stage
        cv2.putText(image, 'STAGE', (150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage if stage else '--', (150, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        # Rep quality stats
        cv2.putText(image, f'Good: {good_reps}', (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Partial: {partial_reps}', (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        # Current ROM
        current_rom = current_rep["max_angle"] - current_rep["min_angle"]
        cv2.putText(image, f'ROM: {int(current_rom)}°', (15, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Form warning
        if form_warning:  # This now works because form_warning is initialized!
            cv2.putText(image, form_warning, (15, image.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display
        cv2.imshow('Bicep Curl Counter - Press Q to Quit', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Final summary
print("\n" + "="*60)
print("WORKOUT SUMMARY")
print("="*60)
print(f"Total Reps: {counter}")
print(f"Good Form Reps: {good_reps}")
print(f"Partial/Incomplete Reps: {partial_reps}")
if counter > 0:
    success_rate = (good_reps / counter) * 100
    print(f"Success Rate: {success_rate:.1f}%")
print("="*60)