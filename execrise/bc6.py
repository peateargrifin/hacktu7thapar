"""
IMPROVED BICEP CURL COUNTER - SEPARATE LEFT & RIGHT ARM TRACKING
Based on analyzed perfect form metrics with more achievable rep quality standards
"""

import cv2
import mediapipe as mp
import numpy as np

# ============================================================================
# METRICS FROM PERFECT FORM VIDEO ANALYSIS - BOTH ARMS
# ============================================================================

LEFT_ARM_METRICS = {
    "down_threshold": 140,        # Much more relaxed (arm doesn't need to be fully extended)
    "up_threshold": 70,            # Much more relaxed (easier to reach curl depth)
    "excellent_peak": 45,          # Very deep curl (relaxed)
    "good_peak": 70,               # Good curl depth (much more achievable)
    "min_rom": 60,                 # Much lower - just need decent movement
    "max_shoulder_drift": 0.050,   # Much more lenient (5x more movement allowed)
}

RIGHT_ARM_METRICS = {
    "down_threshold": 140,        # Much more relaxed (arm doesn't need to be fully extended)
    "up_threshold": 70,            # Much more relaxed (easier to reach curl depth)
    "excellent_peak": 45,          # Very deep curl (relaxed)
    "good_peak": 70,               # Good curl depth (much more achievable)
    "min_rom": 60,                 # Much lower - just need decent movement
    "max_shoulder_drift": 0.050,   # Much more lenient (5x more movement allowed)
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

def calculate_shoulder_drift(current_shoulder, initial_shoulder):
    """Calculate how much the shoulder has moved from initial position"""
    return np.sqrt(
        (current_shoulder[0] - initial_shoulder[0])**2 +
        (current_shoulder[1] - initial_shoulder[1])**2
    )

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
# MAIN BICEP CURL COUNTER
# ============================================================================

# Get target reps from user
print("="*70)
print("BICEP CURL COUNTER - SEPARATE LEFT & RIGHT ARM TRACKING")
print("="*70)
target_reps = input("Enter target reps for EACH arm (e.g., 10): ")
try:
    target_reps = int(target_reps)
except:
    target_reps = 10
    print(f"Invalid input. Using default: {target_reps} reps")

print(f"\nTarget: {target_reps} reps per arm")
print("\n" + "="*70)
print("WORKOUT PLAN:")
print("="*70)
print("1. LEFT ARM CURLS - Complete your set")
print("2. RIGHT ARM CURLS - Complete your set")
print("="*70)
print("\nREP QUALITY STANDARDS (More Achievable!):")
print(f"  • EXCELLENT: Peak curl < {LEFT_ARM_METRICS['excellent_peak']}°, ROM > {LEFT_ARM_METRICS['min_rom']}°")
print(f"  • GOOD: Peak curl < {LEFT_ARM_METRICS['good_peak']}°, ROM > {LEFT_ARM_METRICS['min_rom']}°")
print(f"  • PARTIAL: Doesn't meet good standards but still counted")
print("="*70)
print("\nPress 'Q' to quit | Press 'SPACE' to start/switch arms\n")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

# Workout state
current_arm = "WAITING"  # Start in waiting mode
workout_complete = False

# LEFT ARM tracking
left_counter = 0
left_stage = None
left_initial_shoulder = None
left_good_reps = 0
left_excellent_reps = 0
left_partial_reps = 0
left_current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}

# RIGHT ARM tracking
right_counter = 0
right_stage = None
right_initial_shoulder = None
right_good_reps = 0
right_excellent_reps = 0
right_partial_reps = 0
right_current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}

form_warning = ""

# Initialize display variables (prevent NameError on first frame)
counter = 0
stage = None
good_reps = 0
partial_reps = 0
current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}
elbow_angle = 0

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
        if current_arm == "WAITING":
            # Large prompt to start
            cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (50, 50, 50), -1)
            
            draw_text_with_background(image, "PRESS SPACE TO START LEFT ARM", 
                                    (image.shape[1]//2 - 300, image.shape[0]//2),
                                    font_scale=1.2, thickness=3, 
                                    text_color=(0, 255, 0), bg_color=(0, 0, 0))
            
            draw_text_with_background(image, f"Target: {target_reps} reps per arm", 
                                    (image.shape[1]//2 - 200, image.shape[0]//2 + 60),
                                    font_scale=0.8, thickness=2, 
                                    text_color=(255, 255, 255), bg_color=(0, 0, 0))
            
            cv2.imshow('Bicep Curl Counter - Press SPACE to Start | Q to Quit', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                current_arm = "LEFT"
                print("\n" + "="*70)
                print("✓ STARTING LEFT ARM!")
                print("="*70)
            
            continue

        # ================================================================
        # PROCESS POSE LANDMARKS
        # ================================================================
        try:
            landmarks = results.pose_landmarks.landmark

            # ================================================================
            # PROCESS BASED ON CURRENT ARM
            # ================================================================
            
            if current_arm == "LEFT":
                # Get LEFT arm coordinates
                shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
                
                # Set initial shoulder position
                if left_initial_shoulder is None:
                    left_initial_shoulder = shoulder.copy()
                    print("\n✓ LEFT ARM - Initial position captured!")
                
                # Calculate elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Track min/max for current rep
                if elbow_angle < left_current_rep["min_angle"]:
                    left_current_rep["min_angle"] = elbow_angle
                if elbow_angle > left_current_rep["max_angle"]:
                    left_current_rep["max_angle"] = elbow_angle
                
                # Calculate shoulder drift
                shoulder_drift = calculate_shoulder_drift(shoulder, left_initial_shoulder)
                left_current_rep["shoulder_drift"] = max(left_current_rep["shoulder_drift"], shoulder_drift)
                
                # Reset form warning
                form_warning = ""
                
                # Check shoulder stability
                if shoulder_drift > LEFT_ARM_METRICS["max_shoulder_drift"]:
                    form_warning = "Keep LEFT shoulder stable!"
                
                # BICEP CURL COUNTER LOGIC - LEFT ARM
                if elbow_angle > LEFT_ARM_METRICS["down_threshold"]:
                    left_stage = "down"
                    
                if elbow_angle < LEFT_ARM_METRICS["up_threshold"] and left_stage == 'down':
                    # Rep completed - evaluate form
                    rom = left_current_rep["max_angle"] - left_current_rep["min_angle"]
                    peak_angle = left_current_rep["min_angle"]
                    shoulder_stable = left_current_rep["shoulder_drift"] < LEFT_ARM_METRICS["max_shoulder_drift"]
                    
                    # Determine rep quality - prioritize ROM and be more lenient
                    if rom >= LEFT_ARM_METRICS["min_rom"] and shoulder_stable:
                        # Good ROM and stable shoulders = at least GOOD
                        if peak_angle <= LEFT_ARM_METRICS["excellent_peak"]:
                            form_quality = "⭐ EXCELLENT"
                            left_excellent_reps += 1
                            left_good_reps += 1
                        else:
                            # Any decent curl with good ROM is GOOD
                            form_quality = "✓ GOOD"
                            left_good_reps += 1
                    elif rom >= (LEFT_ARM_METRICS["min_rom"] * 0.75):
                        # 75% of minimum ROM = still good
                        form_quality = "✓ GOOD"
                        left_good_reps += 1
                        if not shoulder_stable:
                            form_quality += " (Watch shoulders)"
                    else:
                        form_quality = "~ PARTIAL"
                        left_partial_reps += 1
                        if rom < LEFT_ARM_METRICS["min_rom"] * 0.5:
                            form_quality += " (Low ROM)"
                        if not shoulder_stable:
                            form_quality += " (Shoulders)"
                    
                    left_stage = "up"
                    left_counter += 1
                    
                    # Print rep summary
                    print(f"LEFT Rep #{left_counter}/{target_reps}: {form_quality} | "
                          f"ROM={rom:.1f}° | Peak={peak_angle:.1f}° | "
                          f"Shoulder Drift={left_current_rep['shoulder_drift']:.3f}")
                    
                    # Reset for next rep
                    left_current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}
                    
                    # Check if left arm set is complete
                    if left_counter >= target_reps:
                        print("\n" + "="*70)
                        print("✓ LEFT ARM SET COMPLETE!")
                        print("="*70)
                        print(f"Total Reps: {left_counter}")
                        print(f"Excellent: {left_excellent_reps} | Good: {left_good_reps - left_excellent_reps} | Partial: {left_partial_reps}")
                        print("="*70)
                        print("\nPress SPACE when ready to start RIGHT ARM")
                        print("="*70)
                        
                        current_arm = "TRANSITION"
                        form_warning = ""
                
                # Current stats for display
                counter = left_counter
                stage = left_stage
                good_reps = left_good_reps
                partial_reps = left_partial_reps
                current_rep = left_current_rep
            
            elif current_arm == "TRANSITION":
                # Show transition screen
                elbow_angle = 0
                counter = left_counter
                stage = "DONE"
                good_reps = left_good_reps
                partial_reps = left_partial_reps
                current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}
                
            elif current_arm == "RIGHT":
                # Get RIGHT arm coordinates
                shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
                
                # Set initial shoulder position
                if right_initial_shoulder is None:
                    right_initial_shoulder = shoulder.copy()
                    print("\n✓ RIGHT ARM - Initial position captured!")
                
                # Calculate elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Track min/max for current rep
                if elbow_angle < right_current_rep["min_angle"]:
                    right_current_rep["min_angle"] = elbow_angle
                if elbow_angle > right_current_rep["max_angle"]:
                    right_current_rep["max_angle"] = elbow_angle
                
                # Calculate shoulder drift
                shoulder_drift = calculate_shoulder_drift(shoulder, right_initial_shoulder)
                right_current_rep["shoulder_drift"] = max(right_current_rep["shoulder_drift"], shoulder_drift)
                
                # Reset form warning
                form_warning = ""
                
                # Check shoulder stability
                if shoulder_drift > RIGHT_ARM_METRICS["max_shoulder_drift"]:
                    form_warning = "Keep RIGHT shoulder stable!"
                
                # BICEP CURL COUNTER LOGIC - RIGHT ARM
                if elbow_angle > RIGHT_ARM_METRICS["down_threshold"]:
                    right_stage = "down"
                    
                if elbow_angle < RIGHT_ARM_METRICS["up_threshold"] and right_stage == 'down':
                    # Rep completed - evaluate form
                    rom = right_current_rep["max_angle"] - right_current_rep["min_angle"]
                    peak_angle = right_current_rep["min_angle"]
                    shoulder_stable = right_current_rep["shoulder_drift"] < RIGHT_ARM_METRICS["max_shoulder_drift"]
                    
                    # Determine rep quality - prioritize ROM and be more lenient
                    if rom >= RIGHT_ARM_METRICS["min_rom"] and shoulder_stable:
                        # Good ROM and stable shoulders = at least GOOD
                        if peak_angle <= RIGHT_ARM_METRICS["excellent_peak"]:
                            form_quality = "⭐ EXCELLENT"
                            right_excellent_reps += 1
                            right_good_reps += 1
                        else:
                            # Any decent curl with good ROM is GOOD
                            form_quality = "✓ GOOD"
                            right_good_reps += 1
                    elif rom >= (RIGHT_ARM_METRICS["min_rom"] * 0.75):
                        # 75% of minimum ROM = still good
                        form_quality = "✓ GOOD"
                        right_good_reps += 1
                        if not shoulder_stable:
                            form_quality += " (Watch shoulders)"
                    else:
                        form_quality = "~ PARTIAL"
                        right_partial_reps += 1
                        if rom < RIGHT_ARM_METRICS["min_rom"] * 0.5:
                            form_quality += " (Low ROM)"
                        if not shoulder_stable:
                            form_quality += " (Shoulders)"
                    
                    right_stage = "up"
                    right_counter += 1
                    
                    # Print rep summary
                    print(f"RIGHT Rep #{right_counter}/{target_reps}: {form_quality} | "
                          f"ROM={rom:.1f}° | Peak={peak_angle:.1f}° | "
                          f"Shoulder Drift={right_current_rep['shoulder_drift']:.3f}")
                    
                    # Reset for next rep
                    right_current_rep = {"min_angle": 180, "max_angle": 0, "shoulder_drift": 0}
                    
                    # Check if right arm set is complete
                    if right_counter >= target_reps:
                        print("\n" + "="*70)
                        print("✓ RIGHT ARM SET COMPLETE!")
                        print("="*70)
                        print(f"Total Reps: {right_counter}")
                        print(f"Excellent: {right_excellent_reps} | Good: {right_good_reps - right_excellent_reps} | Partial: {right_partial_reps}")
                        print("="*70)
                        
                        workout_complete = True
                
                # Current stats for display
                counter = right_counter
                stage = right_stage
                good_reps = right_good_reps
                partial_reps = right_partial_reps
                current_rep = right_current_rep
            
            # ================================================================
            # DISPLAY ANGLE ON ELBOW
            # ================================================================
            if current_arm in ["LEFT", "RIGHT"]:
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

        except Exception as e:
            pass

        # ====================================================================
        # RENDER UI ELEMENTS
        # ====================================================================
        
        if current_arm == "TRANSITION":
            # Transition screen
            cv2.rectangle(image, (0, 0), (image.shape[1], 200), (0, 200, 0), -1)
            
            draw_text_with_background(image, "LEFT ARM COMPLETE!", 
                                    (image.shape[1]//2 - 200, 80),
                                    font_scale=1.5, thickness=3, 
                                    text_color=(255, 255, 255), bg_color=(0, 150, 0))
            
            draw_text_with_background(image, "PRESS SPACE TO START RIGHT ARM", 
                                    (image.shape[1]//2 - 280, 150),
                                    font_scale=1.0, thickness=2, 
                                    text_color=(255, 255, 255), bg_color=(0, 100, 0))
            
            # Show left arm stats
            cv2.putText(image, f"Left Reps: {left_counter} | Good: {left_good_reps} | Partial: {left_partial_reps}", 
                       (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        else:
            # Normal workout UI
            box_color = (66, 135, 245) if current_arm == "LEFT" else (245, 117, 66)
            cv2.rectangle(image, (0, 0), (500, 250), box_color, -1)
            
            # Current arm indicator (BIG)
            cv2.putText(image, f'{current_arm} ARM', (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Rep counter
            cv2.putText(image, 'REPS', (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'{counter}/{target_reps}', (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            # Stage
            cv2.putText(image, 'STAGE', (200, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else '--', (200, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            # Rep quality stats
            cv2.putText(image, f'Good: {good_reps}', (15, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Partial: {partial_reps}', (15, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # Current ROM
            try:
                current_rom = current_rep["max_angle"] - current_rep["min_angle"]
                cv2.putText(image, f'ROM: {int(current_rom)}°', (15, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Progress for other arm (small display)
            if current_arm == "LEFT":
                cv2.putText(image, f'Next: RIGHT (0/{target_reps})', (image.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            elif current_arm == "RIGHT":
                cv2.putText(image, f'Done: LEFT ({left_counter}/{target_reps})', (image.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Form warning
            if form_warning:
                draw_text_with_background(image, form_warning, (15, image.shape[0] - 30),
                            font_scale=0.9, thickness=2, text_color=(0, 0, 255), bg_color=(255, 255, 255))

        # Controls hint
        cv2.putText(image, 'Q: Quit | SPACE: Switch', (image.shape[1] - 320, image.shape[0] - 15),
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
        cv2.imshow('Bicep Curl Counter - Press Q to Quit | SPACE to Switch', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Manual transition from TRANSITION to RIGHT
            if current_arm == "TRANSITION":
                print("\n" + "="*70)
                print("✓ STARTING RIGHT ARM!")
                print("="*70)
                current_arm = "RIGHT"
                form_warning = ""

    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# FINAL WORKOUT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🏋️  WORKOUT COMPLETE - FINAL SUMMARY")
print("="*70)

print("\nLEFT ARM:")
print(f"  Total Reps: {left_counter}/{target_reps}")
print(f"  Excellent: {left_excellent_reps}")
print(f"  Good: {left_good_reps - left_excellent_reps}")
print(f"  Partial: {left_partial_reps}")
if left_counter > 0:
    left_success = (left_good_reps / left_counter) * 100
    print(f"  Quality Rate: {left_success:.1f}%")

print("\nRIGHT ARM:")
print(f"  Total Reps: {right_counter}/{target_reps}")
print(f"  Excellent: {right_excellent_reps}")
print(f"  Good: {right_good_reps - right_excellent_reps}")
print(f"  Partial: {right_partial_reps}")
if right_counter > 0:
    right_success = (right_good_reps / right_counter) * 100
    print(f"  Quality Rate: {right_success:.1f}%")

print("\nTOTAL WORKOUT:")
total_reps = left_counter + right_counter
total_good = left_good_reps + right_good_reps
print(f"  Combined Reps: {total_reps}")
print(f"  Combined Good Reps: {total_good}")
if total_reps > 0:
    overall_success = (total_good / total_reps) * 100
    print(f"  Overall Quality: {overall_success:.1f}%")

print("="*70)
print("Great workout! 💪")
print("="*70)