"""
SIDE LATERAL RAISES COUNTER - BOTH ARMS SIMULTANEOUSLY
Based on analyzed perfect form metrics with achievable rep quality standards
Both arms move together: UP (raise) → DOWN (return)
"""

import cv2
import mediapipe as mp
import numpy as np

# ============================================================================
# METRICS FROM PERFECT FORM VIDEO ANALYSIS - BOTH ARMS
# ============================================================================

LEFT_ARM_METRICS = {
    "down_threshold": 35,          # Relaxed - arms at sides (original: 29.94°)
    "up_threshold": 100,            # Relaxed - arms raised (original: 109.01°)
    "excellent_peak": 105,          # Very good raise (near perfect)
    "good_peak": 95,                # Good raise height
    "min_rom": 60,                  # Minimum range of motion
    "target_elbow": 176,            # Target elbow angle (original: 176.43°)
    "max_elbow_variance": 5,        # Allow some elbow bend variation (original: 2.26°)
    "max_shoulder_elevation": 0.025, # Shrugging detection (original: 0.0199)
}

RIGHT_ARM_METRICS = {
    "down_threshold": 35,          # Relaxed - arms at sides (original: 30.83°)
    "up_threshold": 100,            # Relaxed - arms raised (original: 110.63°)
    "excellent_peak": 105,          # Very good raise
    "good_peak": 95,                # Good raise height
    "min_rom": 60,                  # Minimum range of motion
    "target_elbow": 175,            # Target elbow angle (original: 174.72°)
    "max_elbow_variance": 5,        # Allow some elbow bend variation (original: 1.69°)
    "max_shoulder_elevation": 0.025, # Shrugging detection (original: 0.0222)
}

SYMMETRY_METRICS = {
    "excellent_threshold": 5,       # <5° difference = excellent
    "good_threshold": 10,           # <10° difference = good
    "max_allowed": 15,              # >15° difference = warning
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c where b is the vertex)"""
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

def calculate_shoulder_elevation(current_shoulder, initial_shoulder):
    """Calculate vertical shoulder movement (shrugging detection)"""
    return abs(current_shoulder[1] - initial_shoulder[1])

def draw_text_with_background(image, text, position, font_scale=0.7, thickness=2, 
                               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(image, (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5), bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ============================================================================
# MAIN LATERAL RAISES COUNTER
# ============================================================================

# Get target reps from user
print("="*70)
print("SIDE LATERAL RAISES COUNTER - BOTH ARMS SIMULTANEOUSLY")
print("="*70)
target_reps = input("Enter target reps (e.g., 12): ")
try:
    target_reps = int(target_reps)
except:
    target_reps = 12
    print(f"Invalid input. Using default: {target_reps} reps")

print(f"\nTarget: {target_reps} reps")
print("\n" + "="*70)
print("REP QUALITY STANDARDS:")
print("="*70)
print(f"  • EXCELLENT: Peak > {LEFT_ARM_METRICS['excellent_peak']}°, ROM > {LEFT_ARM_METRICS['min_rom']}°, Symmetry < {SYMMETRY_METRICS['excellent_threshold']}°")
print(f"  • GOOD: Peak > {LEFT_ARM_METRICS['good_peak']}°, ROM > {LEFT_ARM_METRICS['min_rom']}°, Symmetry < {SYMMETRY_METRICS['good_threshold']}°")
print(f"  • PARTIAL: Doesn't meet good standards but still counted")
print("="*70)
print("\nFORM CHECKS:")
print("  ✓ Arms raise together (symmetry)")
print("  ✓ Elbows stay slightly bent (~175°)")
print("  ✓ No shoulder shrugging")
print("  ✓ Full range of motion")
print("="*70)
print("\nPress 'Q' to quit | Press 'SPACE' to start\n")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

# Workout state
workout_active = False
workout_complete = False

# Rep tracking (both arms move together)
counter = 0
stage = None  # "down" or "up"
initial_left_shoulder = None
initial_right_shoulder = None
excellent_reps = 0
good_reps = 0
partial_reps = 0

# Current rep tracking
current_rep = {
    "left_min_angle": 180,
    "left_max_angle": 0,
    "left_elbow_angles": [],
    "left_shoulder_elevation": 0,
    "right_min_angle": 180,
    "right_max_angle": 0,
    "right_elbow_angles": [],
    "right_shoulder_elevation": 0,
    "max_asymmetry": 0,
}

form_warning = ""

# Initialize display variables
left_shoulder_angle = 0
right_shoulder_angle = 0
left_elbow_angle = 0
right_elbow_angle = 0
asymmetry = 0

# Setup MediaPipe Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    while cap.isOpened() and not workout_complete:
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

        # ================================================================
        # HANDLE WAITING STATE
        # ================================================================
        if not workout_active:
            # Large prompt to start
            cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (50, 50, 50), -1)
            
            draw_text_with_background(image, "PRESS SPACE TO START", 
                                    (image.shape[1]//2 - 250, image.shape[0]//2),
                                    font_scale=1.5, thickness=3, 
                                    text_color=(0, 255, 0), bg_color=(0, 0, 0))
            
            draw_text_with_background(image, f"Target: {target_reps} reps | Both arms together", 
                                    (image.shape[1]//2 - 280, image.shape[0]//2 + 60),
                                    font_scale=0.9, thickness=2, 
                                    text_color=(255, 255, 255), bg_color=(0, 0, 0))
            
            draw_text_with_background(image, "Stand with arms at sides to begin", 
                                    (image.shape[1]//2 - 250, image.shape[0]//2 + 120),
                                    font_scale=0.8, thickness=2, 
                                    text_color=(200, 200, 200), bg_color=(0, 0, 0))
            
            cv2.imshow('Lateral Raises Counter - Press SPACE to Start | Q to Quit', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                workout_active = True
                print("\n" + "="*70)
                print("✓ WORKOUT STARTED! Begin your lateral raises...")
                print("="*70)
            
            continue

        # ================================================================
        # PROCESS POSE LANDMARKS (BOTH ARMS)
        # ================================================================
        try:
            landmarks = results.pose_landmarks.landmark
            
            # ============================================================
            # LEFT ARM
            # ============================================================
            left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
            left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
            left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
            
            # Calculate left arm angles
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Store initial shoulder position (first frame)
            if initial_left_shoulder is None:
                initial_left_shoulder = left_shoulder
            
            # Track left shoulder elevation
            left_shoulder_elevation = calculate_shoulder_elevation(left_shoulder, initial_left_shoulder)
            
            # ============================================================
            # RIGHT ARM
            # ============================================================
            right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
            right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
            
            # Calculate right arm angles
            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Store initial shoulder position
            if initial_right_shoulder is None:
                initial_right_shoulder = right_shoulder
            
            # Track right shoulder elevation
            right_shoulder_elevation = calculate_shoulder_elevation(right_shoulder, initial_right_shoulder)
            
            # ============================================================
            # CALCULATE SYMMETRY
            # ============================================================
            asymmetry = abs(left_shoulder_angle - right_shoulder_angle)
            
            # ============================================================
            # TRACK CURRENT REP METRICS
            # ============================================================
            # Left arm
            if left_shoulder_angle < current_rep["left_min_angle"]:
                current_rep["left_min_angle"] = left_shoulder_angle
            if left_shoulder_angle > current_rep["left_max_angle"]:
                current_rep["left_max_angle"] = left_shoulder_angle
            current_rep["left_elbow_angles"].append(left_elbow_angle)
            current_rep["left_shoulder_elevation"] = max(current_rep["left_shoulder_elevation"], left_shoulder_elevation)
            
            # Right arm
            if right_shoulder_angle < current_rep["right_min_angle"]:
                current_rep["right_min_angle"] = right_shoulder_angle
            if right_shoulder_angle > current_rep["right_max_angle"]:
                current_rep["right_max_angle"] = right_shoulder_angle
            current_rep["right_elbow_angles"].append(right_elbow_angle)
            current_rep["right_shoulder_elevation"] = max(current_rep["right_shoulder_elevation"], right_shoulder_elevation)
            
            # Symmetry
            current_rep["max_asymmetry"] = max(current_rep["max_asymmetry"], asymmetry)
            
            # ============================================================
            # REP COUNTING LOGIC (Both arms move together)
            # ============================================================
            # Use average of both arms for stage detection
            avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
            
            # DOWN position (arms at sides)
            if avg_shoulder_angle < LEFT_ARM_METRICS["down_threshold"] and stage == "up":
                # Rep complete! Evaluate quality
                left_rom = current_rep["left_max_angle"] - current_rep["left_min_angle"]
                right_rom = current_rep["right_max_angle"] - current_rep["right_min_angle"]
                avg_rom = (left_rom + right_rom) / 2
                
                left_peak = current_rep["left_max_angle"]
                right_peak = current_rep["right_max_angle"]
                avg_peak = (left_peak + right_peak) / 2
                
                # Check elbow consistency
                left_elbow_std = np.std(current_rep["left_elbow_angles"]) if len(current_rep["left_elbow_angles"]) > 0 else 0
                right_elbow_std = np.std(current_rep["right_elbow_angles"]) if len(current_rep["right_elbow_angles"]) > 0 else 0
                elbow_consistent = (left_elbow_std < LEFT_ARM_METRICS["max_elbow_variance"] and 
                                   right_elbow_std < RIGHT_ARM_METRICS["max_elbow_variance"])
                
                # Check shoulder stability (no shrugging)
                shoulder_stable = (current_rep["left_shoulder_elevation"] < LEFT_ARM_METRICS["max_shoulder_elevation"] and
                                  current_rep["right_shoulder_elevation"] < RIGHT_ARM_METRICS["max_shoulder_elevation"])
                
                # Check symmetry
                symmetry_excellent = current_rep["max_asymmetry"] < SYMMETRY_METRICS["excellent_threshold"]
                symmetry_good = current_rep["max_asymmetry"] < SYMMETRY_METRICS["good_threshold"]
                
                # Determine rep quality
                form_quality = "PARTIAL"
                
                # EXCELLENT rep
                if (avg_peak >= LEFT_ARM_METRICS["excellent_peak"] and 
                    avg_rom >= LEFT_ARM_METRICS["min_rom"] and
                    elbow_consistent and shoulder_stable and symmetry_excellent):
                    form_quality = "EXCELLENT ⭐"
                    excellent_reps += 1
                    good_reps += 1
                
                # GOOD rep
                elif (avg_peak >= LEFT_ARM_METRICS["good_peak"] and 
                      avg_rom >= LEFT_ARM_METRICS["min_rom"] and
                      symmetry_good):
                    form_quality = "GOOD ✓"
                    good_reps += 1
                    if not elbow_consistent:
                        form_quality += " (Elbow bend)"
                    if not shoulder_stable:
                        form_quality += " (Shrugging)"
                
                # PARTIAL rep
                else:
                    partial_reps += 1
                    if avg_rom < LEFT_ARM_METRICS["min_rom"]:
                        form_quality += " (Low ROM)"
                    if not symmetry_good:
                        form_quality += " (Asymmetric)"
                    if not shoulder_stable:
                        form_quality += " (Shrugging)"
                
                stage = "down"
                counter += 1
                
                # Print rep summary
                print(f"Rep #{counter}/{target_reps}: {form_quality} | "
                      f"ROM={avg_rom:.1f}° | Peak={avg_peak:.1f}° | "
                      f"Asymmetry={current_rep['max_asymmetry']:.1f}° | "
                      f"L_elbow_std={left_elbow_std:.1f}° R_elbow_std={right_elbow_std:.1f}°")
                
                # Reset for next rep
                current_rep = {
                    "left_min_angle": 180,
                    "left_max_angle": 0,
                    "left_elbow_angles": [],
                    "left_shoulder_elevation": 0,
                    "right_min_angle": 180,
                    "right_max_angle": 0,
                    "right_elbow_angles": [],
                    "right_shoulder_elevation": 0,
                    "max_asymmetry": 0,
                }
                
                # Check if workout is complete
                if counter >= target_reps:
                    print("\n" + "="*70)
                    print("✓ WORKOUT COMPLETE!")
                    print("="*70)
                    print(f"Total Reps: {counter}")
                    print(f"Excellent: {excellent_reps} | Good: {good_reps - excellent_reps} | Partial: {partial_reps}")
                    print("="*70)
                    workout_complete = True
            
            # UP position (arms raised)
            elif avg_shoulder_angle >= RIGHT_ARM_METRICS["up_threshold"] and stage != "up":
                stage = "up"
            
            # ============================================================
            # FORM WARNINGS (Real-time feedback)
            # ============================================================
            form_warning = ""
            
            # Check for asymmetry
            if asymmetry > SYMMETRY_METRICS["max_allowed"]:
                form_warning = "⚠ UNEVEN ARMS!"
            
            # Check for shrugging
            elif (left_shoulder_elevation > LEFT_ARM_METRICS["max_shoulder_elevation"] or 
                  right_shoulder_elevation > RIGHT_ARM_METRICS["max_shoulder_elevation"]):
                form_warning = "⚠ DON'T SHRUG SHOULDERS!"
            
            # Check for excessive elbow bending
            elif stage == "up":
                avg_elbow = (left_elbow_angle + right_elbow_angle) / 2
                if avg_elbow < 165:
                    form_warning = "⚠ KEEP ELBOWS STRAIGHTER!"
            
            # ============================================================
            # DISPLAY ANGLES ON ARMS
            # ============================================================
            # Left shoulder angle
            cv2.putText(
                image, 
                f'{int(left_shoulder_angle)}°',
                tuple(np.multiply(left_shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Right shoulder angle
            cv2.putText(
                image, 
                f'{int(right_shoulder_angle)}°',
                tuple(np.multiply(right_shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                2, 
                cv2.LINE_AA
            )

        except Exception as e:
            pass

        # ====================================================================
        # RENDER UI ELEMENTS
        # ====================================================================
        
        # Main stats box
        cv2.rectangle(image, (0, 0), (500, 280), (100, 100, 100), -1)
        
        # Rep counter
        cv2.putText(image, 'REPS', (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'{counter}/{target_reps}', (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Stage
        cv2.putText(image, 'STAGE', (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        stage_display = stage.upper() if stage else '--'
        stage_color = (0, 255, 0) if stage == "up" else (255, 255, 255)
        cv2.putText(image, stage_display, (200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, stage_color, 3, cv2.LINE_AA)

        # Rep quality stats
        cv2.putText(image, f'Excellent: {excellent_reps}', (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Good: {good_reps - excellent_reps}', (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Partial: {partial_reps}', (15, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        # Current angles
        cv2.putText(image, f'Left: {int(left_shoulder_angle)}°', (15, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right: {int(right_shoulder_angle)}°', (15, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Asymmetry indicator
        asym_color = (0, 255, 0) if asymmetry < SYMMETRY_METRICS["good_threshold"] else (0, 165, 255) if asymmetry < SYMMETRY_METRICS["max_allowed"] else (0, 0, 255)
        cv2.putText(image, f'Diff: {int(asymmetry)}°', (15, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, asym_color, 2, cv2.LINE_AA)

        # Form warning
        if form_warning:
            draw_text_with_background(image, form_warning, (15, image.shape[0] - 30),
                        font_scale=0.9, thickness=2, text_color=(0, 0, 255), bg_color=(255, 255, 255))

        # Controls hint
        cv2.putText(image, 'Q: Quit', (image.shape[1] - 150, image.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display
        cv2.imshow('Lateral Raises Counter - Press Q to Quit', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# FINAL WORKOUT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🏋️  WORKOUT COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\nTotal Reps: {counter}/{target_reps}")
print(f"  Excellent: {excellent_reps}")
print(f"  Good: {good_reps - excellent_reps}")
print(f"  Partial: {partial_reps}")

if counter > 0:
    quality_rate = (good_reps / counter) * 100
    print(f"\nQuality Rate: {quality_rate:.1f}%")
    
    if quality_rate >= 80:
        print("💪 OUTSTANDING FORM!")
    elif quality_rate >= 60:
        print("✓ Good workout!")
    else:
        print("Keep working on form and symmetry!")

print("="*70)
print("Great workout! 💪")
print("="*70)