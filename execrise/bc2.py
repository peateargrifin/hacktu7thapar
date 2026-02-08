import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# PERFECT FORM REFERENCE DATA (from Denise_Wide-Grip20Curl.mp4)
# These are GUIDANCE values, not strict rules - allow reasonable variation
REFERENCE_METRICS = {
    "starting_angle": 170.7,
    "peak_curl_angle": 28.99,
    "ending_angle": 170.68,
    "full_rom": 143.01,  # Range of Motion
    "max_shoulder_movement": 0.005,
    "avg_shoulder_stability": 0.0032,
    
    # FLEXIBLE Thresholds - focus on motion patterns, not exact values
    "extended_threshold": 140,  # Consider arm "extended" if angle > this (was 155)
    "curled_threshold": 70,  # Consider arm "curled" if angle < this (was 45)
    "min_rom": 60,  # Minimum range of motion - very forgiving (was 110)
    "shoulder_stability_threshold": 0.015,  # Allow more shoulder movement (was 0.008)
}

class BicepCurlCounter:
    def __init__(self):
        self.rep_count = 0
        self.stage = "down"  # "down" or "up"
        self.current_angle = 0
        self.prev_angle = None  # Track previous angle for velocity calculation
        self.min_angle_in_rep = 180
        self.max_angle_in_rep = 0
        self.initial_shoulder_pos = None
        self.shoulder_movement_history = deque(maxlen=30)
        
        # Form feedback
        self.form_issues = []
        self.last_rep_valid = True
        self.rep_start_time = None
        self.rep_times = []
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def get_landmark_coords(self, landmarks, landmark_id):
        """Extract x, y coordinates from landmark"""
        landmark = landmarks.landmark[landmark_id]
        return [landmark.x, landmark.y]
    
    def calculate_shoulder_movement(self, current_shoulder):
        """Calculate how much shoulder has moved from initial position"""
        if self.initial_shoulder_pos is None:
            return 0
        
        movement = np.sqrt(
            (current_shoulder[0] - self.initial_shoulder_pos[0])**2 +
            (current_shoulder[1] - self.initial_shoulder_pos[1])**2
        )
        return movement
    
    def validate_rep(self, rom, avg_shoulder_movement):
        """Validate rep based on motion patterns - focus on trends, not exact values"""
        self.form_issues = []
        valid = True
        
        # Check range of motion - very forgiving, just needs noticeable movement
        if rom < REFERENCE_METRICS["min_rom"]:
            self.form_issues.append(f"Small ROM: {rom:.1f}° (try for >{REFERENCE_METRICS['min_rom']}°)")
            # Still count it if there was some curl motion
            if rom < 40:  # Only reject if extremely small
                valid = False
        
        # Check shoulder stability - warn but don't reject unless excessive
        if avg_shoulder_movement > REFERENCE_METRICS["shoulder_stability_threshold"]:
            self.form_issues.append(f"Watch shoulder stability: {avg_shoulder_movement:.4f}")
            # Only reject if movement is REALLY excessive
            if avg_shoulder_movement > 0.025:
                valid = False
        
        # Provide positive feedback for good form
        if rom > 100:
            self.form_issues.append("Excellent ROM!")
        if avg_shoulder_movement < 0.008:
            self.form_issues.append("Great shoulder stability!")
        
        return valid
    
    def process_frame(self, landmarks):
        """Process a single frame and update rep count"""
        # Get landmark positions (right arm)
        shoulder = self.get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
        elbow = self.get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW)
        wrist = self.get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST)
        
        # Set initial shoulder position
        if self.initial_shoulder_pos is None:
            self.initial_shoulder_pos = shoulder
        
        # Calculate current angle
        self.current_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Calculate angle velocity (direction of movement)
        angle_velocity = 0
        if self.prev_angle is not None:
            angle_velocity = self.current_angle - self.prev_angle
        self.prev_angle = self.current_angle
        
        # Auto-detect initial stage to prevent deadlock if user starts mid-curl
        if self.rep_count == 0 and self.prev_angle is None:
            if self.current_angle < 100:
                self.stage = "up"
                print(f"[AUTO] Starting mid-curl detected - Stage set to UP")
        
        # Track shoulder movement
        shoulder_movement = self.calculate_shoulder_movement(shoulder)
        self.shoulder_movement_history.append(shoulder_movement)
        
        # Track min/max angles in current rep
        if self.current_angle < self.min_angle_in_rep:
            self.min_angle_in_rep = self.current_angle
        if self.current_angle > self.max_angle_in_rep:
            self.max_angle_in_rep = self.current_angle
        
        # DIRECTION-BASED Rep counting logic - detects TRENDS not exact angles
        if self.stage == "down":
            # Detect curl by DIRECTION (decreasing) + entering curl zone
            if angle_velocity < -1 and self.current_angle < 90:
                self.stage = "up"
                if self.rep_start_time is None:
                    self.rep_start_time = time.time()
                print(f"[DEBUG] Curl started | Angle: {self.current_angle:.1f}° | Velocity: {angle_velocity:.2f}")
        
        elif self.stage == "up":
            # Detect extension by DIRECTION (increasing) + extension zone
            if angle_velocity > 1 and self.current_angle > 140:
                # Rep completed - validate it
                rom = self.max_angle_in_rep - self.min_angle_in_rep
                avg_shoulder_movement = np.mean(list(self.shoulder_movement_history)) if self.shoulder_movement_history else 0
                
                print(f"\n[DEBUG] Rep completed!")
                print(f"  ROM: {rom:.1f}° (min: {self.min_angle_in_rep:.1f}°, max: {self.max_angle_in_rep:.1f}°)")
                print(f"  Shoulder movement: {avg_shoulder_movement:.4f}")
                
                self.last_rep_valid = self.validate_rep(rom, avg_shoulder_movement)
                
                if self.last_rep_valid:
                    self.rep_count += 1
                    print(f"  ✓ REP COUNTED! Total: {self.rep_count}")
                    
                    # Track rep time
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                        self.rep_start_time = None
                else:
                    print(f"  ✗ Rep too small")
                
                if self.form_issues:
                    print(f"  Feedback: {', '.join(self.form_issues)}")
                
                # Reset for next rep
                self.stage = "down"
                self.min_angle_in_rep = 180
                self.max_angle_in_rep = 0
                self.shoulder_movement_history.clear()
                print(f"[DEBUG] Ready for next rep\n")
        
        return {
            "angle": self.current_angle,
            "velocity": angle_velocity,
            "stage": self.stage,
            "shoulder_movement": shoulder_movement,
            "rep_count": self.rep_count,
            "form_valid": self.last_rep_valid,
            "form_issues": self.form_issues
        }

def main():
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize counter
    counter = BicepCurlCounter()
    
    # Open webcam (change to video file path if needed)
    # cap = cv2.VideoCapture("path/to/your/video.mp4")
    cap = cv2.VideoCapture(0)  # Use webcam
    
    frame_counter = 0
    
    # Setup pose model
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Pose detection
            results = pose.process(image_rgb)
            
            # Convert back to BGR
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Process landmarks
            if results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
                
                # Process frame and get metrics
                metrics = counter.process_frame(results.pose_landmarks)
                
                # Print angle every 30 frames for monitoring
                if frame_counter % 30 == 0:
                    print(f"Angle: {metrics['angle']:.1f}° | Velocity: {metrics['velocity']:.2f} | Stage: {metrics['stage']} | Reps: {metrics['rep_count']}")
                
                # Display information on screen
                h, w, _ = image_bgr.shape
                
                # Rep count box
                cv2.rectangle(image_bgr, (10, 10), (300, 120), (0, 0, 0), -1)
                cv2.putText(image_bgr, f"REPS: {metrics['rep_count']}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(image_bgr, f"Stage: {metrics['stage'].upper()}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Angle display with velocity
                cv2.rectangle(image_bgr, (10, 130), (350, 220), (0, 0, 0), -1)
                cv2.putText(image_bgr, f"Angle: {metrics['angle']:.1f}", 
                           (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show velocity with color indicator (green=curling, orange=extending, gray=static)
                vel_color = (0, 255, 0) if metrics['velocity'] < -1 else (0, 165, 255) if metrics['velocity'] > 1 else (200, 200, 200)
                cv2.putText(image_bgr, f"Vel: {metrics['velocity']:.1f}", 
                           (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vel_color, 2)
                
                # Form feedback - show guidance, not just pass/fail
                feedback_color = (0, 200, 0) if metrics['form_valid'] else (0, 165, 255)  # Green or Orange
                cv2.rectangle(image_bgr, (10, 230), (450, 280), feedback_color, -1)
                
                if metrics['form_valid'] and not metrics['form_issues']:
                    cv2.putText(image_bgr, "FORM: EXCELLENT!", 
                               (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif metrics['form_valid']:
                    cv2.putText(image_bgr, "FORM: GOOD - See tips", 
                               (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    cv2.putText(image_bgr, "Keep working on form!", 
                               (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Display form feedback/tips
                if metrics['form_issues']:
                    y_offset = 300
                    color = (0, 255, 0) if metrics['form_valid'] else (0, 165, 255)
                    for i, issue in enumerate(metrics['form_issues'][:3]):  # Show max 3 tips
                        cv2.putText(image_bgr, f"• {issue}", 
                                   (20, y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 2)
                
                # Progress bar for angle
                angle_normalized = int((metrics['angle'] / 180) * 400)
                cv2.rectangle(image_bgr, (w-50, h-angle_normalized), (w-20, h), (0, 255, 0), -1)
                
            else:
                # No person detected
                cv2.putText(image_bgr, "NO PERSON DETECTED", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Bicep Curl Counter - Perfect Form Tracker", image_bgr)
            
            # Press Q to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*60)
    print("WORKOUT SUMMARY")
    print("="*60)
    print(f"Total Valid Reps: {counter.rep_count}")
    if counter.rep_times:
        print(f"Average Rep Time: {np.mean(counter.rep_times):.2f} seconds")
        print(f"Fastest Rep: {min(counter.rep_times):.2f} seconds")
        print(f"Slowest Rep: {max(counter.rep_times):.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()