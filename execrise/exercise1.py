# -*- coding: utf-8 -*-
"""
Enhanced Bicep Curl Counter with Form Feedback
Features:
- Select target reps for each hand
- Form feedback and improvement messages
- Hand switching prompts
- Total time tracking
- Performance summary
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime, timedelta

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=20):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    # Draw circles at corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def get_feedback_message(angle, stage, last_angle):
    """Provide form feedback based on curl execution"""
    messages = []
    
    if stage == "down":
        if angle < 140:
            messages.append("Extend arm fully!")
        elif angle >= 140 and angle < 160:
            messages.append("Almost there - extend more")
        else:
            messages.append("Good extension!")
    
    elif stage == "up":
        if angle > 50:
            messages.append("Curl higher!")
        elif angle >= 30 and angle <= 50:
            messages.append("Good curl - control it!")
        else:
            messages.append("Perfect form!")
    
    # Check for smooth movement
    if last_angle is not None:
        angle_change = abs(angle - last_angle)
        if angle_change > 20:
            messages.append("Move slower - control!")
    
    return messages


def format_time(seconds):
    """Format seconds into MM:SS"""
    return str(timedelta(seconds=int(seconds)))[2:]


def main():
    # Setup
    print("=" * 60)
    print("BICEP CURL COUNTER - Enhanced Edition")
    print("=" * 60)
    
    # Get target reps for each hand
    while True:
        try:
            left_target = int(input("\nEnter target reps for LEFT hand: "))
            right_target = int(input("Enter target reps for RIGHT hand: "))
            if left_target > 0 and right_target > 0:
                break
            print("Please enter positive numbers!")
        except ValueError:
            print("Please enter valid numbers!")
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("- Position yourself so your full arm is visible")
    print("- Extend arm fully (angle > 160°) for 'down' position")
    print("- Curl to angle < 30° for 'up' position")
    print("- Press 'q' to quit, 's' to switch hands manually")
    print("=" * 60)
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Counter variables
    left_counter = 0
    right_counter = 0
    current_hand = "LEFT"
    stage = None
    last_angle = None
    feedback_messages = []
    feedback_timer = 0
    
    # Performance tracking
    left_start_time = None
    right_start_time = None
    left_end_time = None
    right_end_time = None
    overall_start_time = time.time()
    
    # Hand switch tracking
    hand_switch_prompted = False
    
    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get image dimensions
            h, w, _ = image.shape
            
            # Set default counter and target based on current hand
            if current_hand == "LEFT":
                counter = left_counter
                target = left_target
            else:
                counter = right_counter
                target = right_target
            
            # Extract landmarks and process
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Select landmarks based on current hand
                if current_hand == "LEFT":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Start timer for left hand
                    if left_start_time is None and left_counter == 0:
                        left_start_time = time.time()
                else:
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    # Start timer for right hand
                    if right_start_time is None and right_counter == 0:
                        right_start_time = time.time()
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Get feedback
                feedback_messages = get_feedback_message(angle, stage, last_angle)
                feedback_timer = time.time()
                
                # Visualize angle on elbow
                elbow_coords = tuple(np.multiply(elbow, [w, h]).astype(int))
                cv2.putText(image, f"{int(angle)}°",
                           (elbow_coords[0] - 30, elbow_coords[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    
                    if current_hand == "LEFT":
                        left_counter += 1
                        counter = left_counter  # Update display counter
                        
                        # Check if left hand is complete
                        if left_counter >= left_target and left_end_time is None:
                            left_end_time = time.time()
                            feedback_messages = ["LEFT HAND COMPLETE! 🎉", "Switch to RIGHT hand"]
                            hand_switch_prompted = True
                    else:
                        right_counter += 1
                        counter = right_counter  # Update display counter
                        
                        # Check if right hand is complete
                        if right_counter >= right_target and right_end_time is None:
                            right_end_time = time.time()
                            feedback_messages = ["RIGHT HAND COMPLETE! 🎉", "Great job!"]
                
                last_angle = angle
                
            except Exception as e:
                feedback_messages = ["Position yourself in frame!", "Show full arm"]
            
            # Create UI overlay
            overlay = image.copy()
            
            # Main status panel (top)
            draw_rounded_rectangle(overlay, (20, 20), (w - 20, 180), (40, 40, 40), -1, 15)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Current hand indicator
            hand_color = (100, 255, 100) if current_hand == "LEFT" else (100, 150, 255)
            cv2.putText(image, f"CURRENT: {current_hand} HAND",
                       (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, hand_color, 3, cv2.LINE_AA)
            
            # Reps counter
            progress = min(counter / target, 1.0)
            counter_color = (100, 255, 100) if counter >= target else (255, 255, 255)
            cv2.putText(image, f"REPS: {counter}/{target}",
                       (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, counter_color, 3, cv2.LINE_AA)
            
            # Progress bar
            bar_width = 400
            bar_x = 40
            bar_y = 130
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (60, 60, 60), -1)
            cv2.rectangle(image, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + 20), hand_color, -1)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 2)
            
            # Stage indicator
            stage_text = stage if stage else "READY"
            stage_color = (100, 255, 100) if stage == "up" else (255, 200, 100)
            cv2.putText(image, f"STAGE: {stage_text}",
                       (w - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, stage_color, 2, cv2.LINE_AA)
            
            # Time tracking
            elapsed = time.time() - overall_start_time
            cv2.putText(image, f"TIME: {format_time(elapsed)}",
                       (w - 350, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2, cv2.LINE_AA)
            
            # Both hands status (side panel)
            panel_x = w - 300
            overlay2 = image.copy()
            draw_rounded_rectangle(overlay2, (panel_x, 200), (w - 20, 450), (40, 40, 40), -1, 15)
            cv2.addWeighted(overlay2, 0.7, image, 0.3, 0, image)
            
            # Left hand status
            left_status_color = (100, 255, 100) if left_counter >= left_target else (255, 255, 255)
            cv2.putText(image, "LEFT HAND",
                       (panel_x + 20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(image, f"{left_counter}/{left_target} reps",
                       (panel_x + 20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_status_color, 2, cv2.LINE_AA)
            
            if left_end_time and left_start_time:
                left_time = left_end_time - left_start_time
                cv2.putText(image, f"Time: {format_time(left_time)}",
                           (panel_x + 20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Right hand status
            right_status_color = (100, 255, 100) if right_counter >= right_target else (255, 255, 255)
            cv2.putText(image, "RIGHT HAND",
                       (panel_x + 20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{right_counter}/{right_target} reps",
                       (panel_x + 20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_status_color, 2, cv2.LINE_AA)
            
            if right_end_time and right_start_time:
                right_time = right_end_time - right_start_time
                cv2.putText(image, f"Time: {format_time(right_time)}",
                           (panel_x + 20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Feedback messages (bottom)
            if feedback_messages and (time.time() - feedback_timer < 2):
                overlay3 = image.copy()
                msg_height = 80 + (len(feedback_messages) * 40)
                draw_rounded_rectangle(overlay3, (20, h - msg_height - 20), 
                                      (w - 20, h - 20), (40, 40, 40), -1, 15)
                cv2.addWeighted(overlay3, 0.8, image, 0.2, 0, image)
                
                y_offset = h - msg_height + 20
                for i, msg in enumerate(feedback_messages):
                    msg_color = (100, 255, 100) if "Good" in msg or "Perfect" in msg or "COMPLETE" in msg else (255, 200, 100)
                    cv2.putText(image, msg,
                               (40, y_offset + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, msg_color, 2, cv2.LINE_AA)
            
            # Hand switch prompt
            if left_counter >= left_target and current_hand == "LEFT":
                if not hand_switch_prompted or (time.time() - feedback_timer > 2):
                    overlay4 = image.copy()
                    draw_rounded_rectangle(overlay4, (w//2 - 300, h//2 - 80), 
                                         (w//2 + 300, h//2 + 80), (50, 200, 50), -1, 20)
                    cv2.addWeighted(overlay4, 0.9, image, 0.1, 0, image)
                    
                    cv2.putText(image, "LEFT HAND COMPLETE!",
                               (w//2 - 280, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(image, "Press 'S' to switch to RIGHT hand",
                               (w//2 - 280, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render pose detections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Show the image
            cv2.imshow('Enhanced Bicep Curl Counter', image)
            
            # Key controls
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') or key == ord('S'):
                # Manual hand switch
                if current_hand == "LEFT":
                    current_hand = "RIGHT"
                    stage = None
                    hand_switch_prompted = False
                    feedback_messages = ["Switched to RIGHT hand!", "Let's go!"]
                    feedback_timer = time.time()
                elif current_hand == "RIGHT":
                    current_hand = "LEFT"
                    stage = None
                    hand_switch_prompted = False
                    feedback_messages = ["Switched to LEFT hand!", "Let's go!"]
                    feedback_timer = time.time()
            
            # Auto-complete check
            if left_counter >= left_target and right_counter >= right_target:
                # Both hands complete
                overlay5 = image.copy()
                draw_rounded_rectangle(overlay5, (w//2 - 350, h//2 - 120), 
                                     (w//2 + 350, h//2 + 120), (50, 200, 50), -1, 20)
                cv2.addWeighted(overlay5, 0.9, image, 0.1, 0, image)
                
                cv2.putText(image, "WORKOUT COMPLETE! 🎉",
                           (w//2 - 300, h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.8, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(image, "Press 'Q' to finish",
                           (w//2 - 200, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('Enhanced Bicep Curl Counter', image)
                cv2.waitKey(3000)  # Show for 3 seconds
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print("\n" + "=" * 60)
    print("WORKOUT SUMMARY")
    print("=" * 60)
    print(f"\nLeft Hand:  {left_counter}/{left_target} reps", end="")
    if left_end_time and left_start_time:
        left_duration = left_end_time - left_start_time
        print(f" - Time: {format_time(left_duration)}")
    else:
        print()
    
    print(f"Right Hand: {right_counter}/{right_target} reps", end="")
    if right_end_time and right_start_time:
        right_duration = right_end_time - right_start_time
        print(f" - Time: {format_time(right_duration)}")
    else:
        print()
    
    print(f"\nTotal Time: {format_time(total_time)}")
    print(f"Total Reps: {left_counter + right_counter}")
    
    # Performance feedback
    print("\n" + "=" * 60)
    print("PERFORMANCE FEEDBACK")
    print("=" * 60)
    
    if left_counter >= left_target and right_counter >= right_target:
        print("✓ Excellent! You completed all target reps!")
    elif left_counter >= left_target or right_counter >= right_target:
        print("✓ Good effort! You completed one hand.")
        print("  Try to complete both hands next time!")
    else:
        print("  Keep practicing! Consistency is key.")
    
    if left_end_time and left_start_time and right_end_time and right_start_time:
        left_duration = left_end_time - left_start_time
        right_duration = right_end_time - right_start_time
        
        if abs(left_duration - right_duration) < 5:
            print("✓ Great balance! Both hands took similar time.")
        else:
            slower_hand = "LEFT" if left_duration > right_duration else "RIGHT"
            print(f"  {slower_hand} hand took longer. Work on building")
            print(f"  strength and endurance on that side.")
    
    print("\nTips for next session:")
    print("• Maintain controlled movements")
    print("• Fully extend arm at bottom (>160°)")
    print("• Full curl at top (<30°)")
    print("• Keep consistent tempo")
    print("=" * 60)
    print("\nThank you for using Enhanced Bicep Curl Counter!")
    print("=" * 60)


if __name__ == "__main__":
    main()