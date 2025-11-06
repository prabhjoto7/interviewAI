
"""
Real-Time Interview Recording and Violation Detection System
UPDATED VERSION:
- Fixed cv2.FONT_HERSHEY_BOLD error (use FONT_HERSHEY_SIMPLEX)
- Captures violation images
- Continues to next question after violation
- Stores violation metadata for display in results
"""

import cv2
import numpy as np
import threading
import time
import tempfile
import os
import speech_recognition as sr
import warnings
from collections import deque

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RecordingSystem:
    """Handles video/audio recording with real-time violation detection"""
    
    def __init__(self, models_dict):
        """
        Initialize recording system with loaded models
        
        Args:
            models_dict: Dictionary containing pre-loaded AI models
        """
        self.models = models_dict
        self.violation_detected = False
        self.violation_reason = ""
        
        # Frame boundaries (for sitting position: left, right, top only)
        self.frame_margin = 50  # pixels from edge
        
        # Position adjustment tracking
        self.position_adjusted = False
        self.baseline_environment = None  # Store initial environment scan
        
        # Violation storage directory
        self.violation_images_dir = tempfile.mkdtemp(prefix="violations_")
        
        # Initialize pose detection if available
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_available = True
        except:
            self.pose_detector = None
            self.pose_available = False
    
    def save_violation_image(self, frame, question_number, violation_reason):
        """
        Save an image of the violation for later display
        FIXED: Changed cv2.FONT_HERSHEY_BOLD to cv2.FONT_HERSHEY_SIMPLEX
        
        Args:
            frame: BGR image frame showing the violation
            question_number: Current question number
            violation_reason: Description of the violation
            
        Returns:
            Path to saved violation image
        """
        try:
            # Create filename with timestamp
            timestamp = int(time.time() * 1000)
            filename = f"violation_q{question_number}_{timestamp}.jpg"
            filepath = os.path.join(self.violation_images_dir, filename)
            
            # Add violation text overlay to image
            overlay_frame = frame.copy()
            h, w = overlay_frame.shape[:2]
            
            # Add semi-transparent red overlay
            red_overlay = overlay_frame.copy()
            cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, red_overlay, 0.3, 0)
            
            # Add thick red border
            cv2.rectangle(overlay_frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
            # Add violation text with background - FIXED FONT
            text = "VIOLATION DETECTED"
            cv2.rectangle(overlay_frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(overlay_frame, text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # FIXED: Was FONT_HERSHEY_BOLD
            
            # Add violation reason at bottom
            cv2.rectangle(overlay_frame, (0, h-100), (w, h), (0, 0, 0), -1)
            
            # Split long violation text into multiple lines
            words = violation_reason.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 50:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            # Draw violation reason lines
            y_offset = h - 90
            for line in lines[:2]:  # Max 2 lines
                cv2.putText(overlay_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Save image
            cv2.imwrite(filepath, overlay_frame)
            return filepath
            
        except Exception as e:
            print(f"Error saving violation image: {e}")
            return None
    
    def scan_environment(self, frame):
        """
        Scan and catalog the environment before test starts
        """
        if self.models['yolo'] is None:
            return {'objects': [], 'positions': []}
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
            environment_data = {
                'objects': [],
                'positions': [],
                'person_position': None
            }
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    environment_data['objects'].append(obj_name)
                    environment_data['positions'].append({
                        'name': obj_name,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1+x2)/2), int((y1+y2)/2))
                    })
                    
                    if obj_name == 'person':
                        environment_data['person_position'] = (int((x1+x2)/2), int((y1+y2)/2))
            
            return environment_data
            
        except Exception as e:
            return {'objects': [], 'positions': []}
    
    def detect_new_objects(self, frame):
        """
        Detect NEW objects that weren't in baseline environment
        """
        if self.models['yolo'] is None or self.baseline_environment is None:
            return False, []
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                current_objects = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    current_center = (int((x1+x2)/2), int((y1+y2)/2))
                    
                    current_objects.append({
                        'name': obj_name,
                        'center': current_center,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                
                baseline_objects = self.baseline_environment['positions']
                new_items = []
                
                for curr_obj in current_objects:
                    if curr_obj['name'] == 'person':
                        continue
                    
                    is_baseline = False
                    for base_obj in baseline_objects:
                        if curr_obj['name'] == base_obj['name']:
                            dist = np.sqrt(
                                (curr_obj['center'][0] - base_obj['center'][0])**2 +
                                (curr_obj['center'][1] - base_obj['center'][1])**2
                            )
                            if dist < 100:
                                is_baseline = True
                                break
                    
                    if not is_baseline:
                        new_items.append(curr_obj['name'])
                
                if new_items:
                    return True, list(set(new_items))
            
            return False, []
            
        except Exception as e:
            return False, []
    
    def detect_suspicious_movements(self, frame):
        """Detect suspicious hand movements"""
        if self.models['hands'] is None:
            return False, ""
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        try:
            hand_results = self.models['hands'].process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    
                    wrist_y = wrist.y * h
                    tip_y = index_tip.y * h
                    
                    if wrist_y > h * 0.75:
                        return True, "Hand movement below desk level detected"
                    
                    if wrist_y < h * 0.15:
                        return True, "Suspicious hand movement at top of frame"
        
        except Exception as e:
            pass
        
        return False, ""
    
    def calculate_eye_gaze(self, face_landmarks, frame_shape):
        """Calculate if eyes are looking at camera"""
        h, w = frame_shape[:2]
        
        left_eye_indices = [468, 469, 470, 471, 472]
        right_eye_indices = [473, 474, 475, 476, 477]
        left_eye_center = [33, 133, 157, 158, 159, 160, 161, 163, 144, 145, 153, 154, 155]
        right_eye_center = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 373, 374, 390]
        
        landmarks = face_landmarks.landmark
        
        left_iris_x = np.mean([landmarks[i].x for i in left_eye_indices if i < len(landmarks)])
        left_eye_x = np.mean([landmarks[i].x for i in left_eye_center if i < len(landmarks)])
        
        right_iris_x = np.mean([landmarks[i].x for i in right_eye_indices if i < len(landmarks)])
        right_eye_x = np.mean([landmarks[i].x for i in right_eye_center if i < len(landmarks)])
        
        left_gaze_ratio = (left_iris_x - left_eye_x) if left_iris_x and left_eye_x else 0
        right_gaze_ratio = (right_iris_x - right_eye_x) if right_iris_x and right_eye_x else 0
        
        avg_gaze = (left_gaze_ratio + right_gaze_ratio) / 2
        
        return abs(avg_gaze) < 0.02
    
    def estimate_head_pose(self, face_landmarks, frame_shape):
        """Estimate head pose angles"""
        h, w = frame_shape[:2]
        landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark])
        
        required_indices = [1, 33, 263, 61, 291]
        image_points = np.array([landmarks_3d[i] for i in required_indices], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -125.0, -30.0),
            (30.0, -125.0, -30.0), (-60.0, -70.0, -60.0),
            (60.0, -70.0, -60.0)
        ])
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, 
            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rmat, rotation_vector))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
            yaw, pitch, roll = [float(a) for a in euler]
            return yaw, pitch, roll
        
        return 0, 0, 0
    
    def detect_blink(self, face_landmarks):
        """Detect if eye is blinking"""
        upper_lid = face_landmarks.landmark[159]
        lower_lid = face_landmarks.landmark[145]
        eye_openness = abs(upper_lid.y - lower_lid.y)
        return eye_openness < 0.01
    
    def analyze_lighting(self, frame):
        """Analyze lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if mean_brightness < 60:
            return "Too Dark", mean_brightness
        elif mean_brightness > 200:
            return "Too Bright", mean_brightness
        elif std_brightness < 25:
            return "Low Contrast", mean_brightness
        else:
            return "Good", mean_brightness
    
    
    def check_frame_boundaries(self, frame, face_box):
        """Check if person is within frame boundaries"""
        if face_box is None:
            return False, "No face detected", "NO_FACE"
        
        h, w = frame.shape[:2]
        margin = self.frame_margin
        x, y, fw, fh = face_box
        
        face_center_x = x + fw // 2
        face_top = y
        face_left = x
        face_right = x + fw
        
        if face_left < margin:
            return False, "Person too close to LEFT edge", "LEFT_VIOLATION"
        
        if face_right > (w - margin):
            return False, "Person too close to RIGHT edge", "RIGHT_VIOLATION"
        
        if face_top < margin:
            return False, "Person too close to TOP edge", "TOP_VIOLATION"
        
        return True, "Within boundaries", "OK"
    
    def detect_person_outside_frame(self, frame):
        """Detect if any person/living being is outside boundaries"""
        if self.models['yolo'] is None:
            return False, "", ""
        
        h, w = frame.shape[:2]
        margin = self.frame_margin
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                living_beings = ['person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 
                                'elephant', 'bear', 'zebra', 'giraffe']
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    
                    if obj_name.lower() in living_beings:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if x1 < margin or x2 < margin:
                            return True, obj_name, "LEFT"
                        
                        if x1 > (w - margin) or x2 > (w - margin):
                            return True, obj_name, "RIGHT"
                        
                        if y1 < margin or y2 < margin:
                            return True, obj_name, "TOP"
        
        except Exception as e:
            pass
        
        return False, "", ""
    
    def detect_multiple_bodies(self, frame, num_faces):
        """Detect multiple bodies using pose and hand detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        body_count = 0
        detected_parts = []
        
        if self.pose_available and self.pose_detector:
            try:
                pose_results = self.pose_detector.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    body_count += 1
                    detected_parts.append("body")
                    
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    visible_shoulders = sum(1 for idx in [11, 12] 
                                          if landmarks[idx].visibility > 0.5)
                    visible_elbows = sum(1 for idx in [13, 14] 
                                        if landmarks[idx].visibility > 0.5)
                    
                    if visible_shoulders > 2 or visible_elbows > 2:
                        return True, "Multiple body parts detected (extra shoulders/arms)", body_count + 1
                        
            except Exception as e:
                pass
        
        if self.models['hands'] is not None:
            try:
                hand_results = self.models['hands'].process(rgb_frame)
                
                if hand_results.multi_hand_landmarks:
                    num_hands = len(hand_results.multi_hand_landmarks)
                    
                    if num_hands > 2:
                        detected_parts.append(f"{num_hands} hands")
                        return True, f"Multiple persons detected ({num_hands} hands visible)", 2
                    
                    if num_hands == 2:
                        hand1 = hand_results.multi_hand_landmarks[0].landmark[0]
                        hand2 = hand_results.multi_hand_landmarks[1].landmark[0]
                        
                        distance = np.sqrt((hand1.x - hand2.x)**2 + (hand1.y - hand2.y)**2)
                        
                        if distance > 0.7:
                            detected_parts.append("widely separated hands")
                            return True, "Suspicious hand positions (possible multiple persons)", 2
                            
            except Exception as e:
                pass
        
        if num_faces == 1 and body_count > 1:
            return True, "Body parts from multiple persons detected", 2
        
        if num_faces > 1:
            return True, f"Multiple persons detected ({num_faces} faces)", num_faces
        
        return False, "", max(num_faces, body_count)
    
    def detect_hands_outside_main_person(self, frame, face_box):
        """Detect hands outside main person's area"""
        if self.models['hands'] is None or face_box is None:
            return False, ""
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        try:
            hand_results = self.models['hands'].process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                x, y, fw, fh = face_box
                
                expected_left = max(0, x - fw)
                expected_right = min(w, x + fw * 2)
                expected_top = max(0, y - fh)
                expected_bottom = min(h, y + fh * 4)
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_x = hand_landmarks.landmark[0].x * w
                    hand_y = hand_landmarks.landmark[0].y * h
                    
                    if (hand_x < expected_left - 50 or hand_x > expected_right + 50 or
                        hand_y < expected_top - 50 or hand_y > expected_bottom + 50):
                        return True, "Hand detected outside main person's area"
                
        except Exception as e:
            pass
        
        return False, ""
    
    def has_skin_tone(self, region):
        """Check if region contains skin-like colors"""
        if region.size == 0:
            return False
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([0, 20, 0], dtype=np.uint8)
        upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        skin_ratio = np.sum(mask > 0) / mask.size
        return skin_ratio > 0.3
    
    def detect_intrusion_at_edges(self, frame, face_box):
        """Detect body parts intruding from frame edges"""
        if face_box is None:
            return False, ""
        
        h, w = frame.shape[:2]
        x, y, fw, fh = face_box
        
        edge_width = 80
        
        left_region = frame[:, :edge_width]
        right_region = frame[:, w-edge_width:]
        top_left = frame[:edge_width, :w//3]
        top_right = frame[:edge_width, 2*w//3:]
        
        face_center_x = x + fw // 2
        face_far_from_left = face_center_x > w * 0.3
        face_far_from_right = face_center_x < w * 0.7
        
        if face_far_from_left and self.has_skin_tone(left_region):
            if self.models['hands']:
                rgb_region = cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB)
                try:
                    result = self.models['hands'].process(rgb_region)
                    if result.multi_hand_landmarks:
                        return True, "Body part detected at left edge (another person)"
                except:
                    pass
        
        if face_far_from_right and self.has_skin_tone(right_region):
            if self.models['hands']:
                rgb_region = cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB)
                try:
                    result = self.models['hands'].process(rgb_region)
                    if result.multi_hand_landmarks:
                        return True, "Body part detected at right edge (another person)"
                except:
                    pass
        
        if y > h * 0.2:
            if self.has_skin_tone(top_left) or self.has_skin_tone(top_right):
                return True, "Body part detected at top edge (another person)"
        
        return False, ""
    
    def draw_frame_boundaries(self, frame):
        """Draw visible frame boundaries"""
        h, w = frame.shape[:2]
        margin = self.frame_margin
        
        overlay = frame.copy()
        
        cv2.line(overlay, (margin, 0), (margin, h), (0, 255, 0), 3)
        cv2.line(overlay, (w - margin, 0), (w - margin, h), (0, 255, 0), 3)
        cv2.line(overlay, (0, margin), (w, margin), (0, 255, 0), 3)
        cv2.rectangle(overlay, (margin, margin), (w - margin, h), (0, 255, 0), 2)
        
        frame_with_boundaries = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        corner_size = 30
        cv2.line(frame_with_boundaries, (margin, margin), (margin + corner_size, margin), (0, 255, 0), 3)
        cv2.line(frame_with_boundaries, (margin, margin), (margin, margin + corner_size), (0, 255, 0), 3)
        
        cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin - corner_size, margin), (0, 255, 0), 3)
        cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin, margin + corner_size), (0, 255, 0), 3)
        
        cv2.putText(frame_with_boundaries, "Stay within GREEN boundaries", 
                    (w//2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_with_boundaries
    
    def pre_test_setup_phase(self, ui_callbacks, timeout=60):
        """
        ONE-TIME pre-test setup phase with environment scanning
        """
        if self.position_adjusted:
            return True
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        start_time = time.time()
        position_ok_counter = 0
        required_stable_frames = 30
        
        ui_callbacks['countdown_update']("📸 ONE-TIME SETUP: Adjust your position within the GREEN frame")
        
        while (time.time() - start_time) < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            frame_with_boundaries = self.draw_frame_boundaries(frame)
            
            face_box = None
            is_ready = False
            status_message = "Detecting face..."
            status_color = (255, 165, 0)
            
            if self.models['face_mesh'] is not None:
                face_results = self.models['face_mesh'].process(rgb_frame)
                
                if face_results.multi_face_landmarks:
                    num_faces = len(face_results.multi_face_landmarks)
                    
                    if num_faces > 1:
                        status_message = "⚠️ Multiple faces detected! Only ONE person allowed"
                        status_color = (0, 0, 255)
                        position_ok_counter = 0
                    
                    elif num_faces == 1:
                        face_landmarks = face_results.multi_face_landmarks[0]
                        
                        landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                        x_coords = landmarks_2d[:, 0]
                        y_coords = landmarks_2d[:, 1]
                        face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
                                   int(np.max(x_coords) - np.min(x_coords)), 
                                   int(np.max(y_coords) - np.min(y_coords)))
                        
                        within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, face_box)
                        
                        outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
                        
                        if outside_detected:
                            status_message = f"⚠️ {obj_type.upper()} detected outside frame ({location} side)!"
                            status_color = (0, 0, 255)
                            position_ok_counter = 0
                        
                        elif not within_bounds:
                            status_message = f"⚠️ {boundary_msg} - Please adjust!"
                            status_color = (0, 0, 255)
                            position_ok_counter = 0
                            
                            if boundary_status == "LEFT_VIOLATION":
                                cv2.rectangle(frame_with_boundaries, (0, 0), (self.frame_margin, h), (0, 0, 255), -1)
                            elif boundary_status == "RIGHT_VIOLATION":
                                cv2.rectangle(frame_with_boundaries, (w - self.frame_margin, 0), (w, h), (0, 0, 255), -1)
                            elif boundary_status == "TOP_VIOLATION":
                                cv2.rectangle(frame_with_boundaries, (0, 0), (w, self.frame_margin), (0, 0, 255), -1)
                        
                        else:
                            position_ok_counter += 1
                            progress = min(100, int((position_ok_counter / required_stable_frames) * 100))
                            status_message = f"✅ Good position! Hold steady... {progress}%"
                            status_color = (0, 255, 0)
                            
                            if position_ok_counter >= required_stable_frames:
                                is_ready = True
                
                else:
                    status_message = "❌ No face detected - Please position yourself in frame"
                    status_color = (0, 0, 255)
                    position_ok_counter = 0
            
            overlay_height = 140
            overlay = frame_with_boundaries.copy()
            cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
            frame_with_boundaries = cv2.addWeighted(frame_with_boundaries, 0.7, overlay, 0.3, 0)
            
            cv2.putText(frame_with_boundaries, status_message, (10, h - 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.putText(frame_with_boundaries, "Instructions:", (10, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_with_boundaries, "• Keep your face within GREEN boundaries", (10, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_with_boundaries, "• Ensure no one else is visible", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_with_boundaries, "• Remove all unauthorized items from view", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            ui_callbacks['video_update'](cv2.resize(frame_with_boundaries, (640, 480)))
            
            elapsed = int(time.time() - start_time)
            ui_callbacks['timer_update'](f"⏱️ Setup time: {elapsed}s / {timeout}s")
            
            if is_ready:
                ui_callbacks['countdown_update']("🔍 Scanning environment... Please stay still")
                time.sleep(1)
                
                baseline_frames = []
                for _ in range(10):
                    ret, scan_frame = cap.read()
                    if ret:
                        baseline_frames.append(scan_frame)
                    time.sleep(0.1)
                
                if baseline_frames:
                    self.baseline_environment = self.scan_environment(baseline_frames[len(baseline_frames)//2])
                
                success_frame = frame_with_boundaries.copy()
                cv2.rectangle(success_frame, (0, 0), (w, h), (0, 255, 0), 10)
                cv2.putText(success_frame, "SETUP COMPLETE!", 
                           (w//2 - 180, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(success_frame, "Test will begin shortly...", 
                           (w//2 - 180, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                ui_callbacks['video_update'](cv2.resize(success_frame, (640, 480)))
                time.sleep(3)
                
                cap.release()
                ui_callbacks['countdown_update']('')
                self.position_adjusted = True
                return True
            
            time.sleep(0.03)
        
        cap.release()
        ui_callbacks['countdown_update']('⚠️ Setup timeout - Please try again')
        return False
    
    def record_interview(self, question_data, duration, ui_callbacks):
        """
        DEPRECATED: Use record_continuous_interview() instead
        Kept for backward compatibility
        """
        result = self.record_continuous_interview([question_data], duration, ui_callbacks)
        
        if isinstance(result, dict) and 'questions_results' in result:
            if result['questions_results']:
                first_result = result['questions_results'][0]
                first_result['video_path'] = result.get('session_video_path', '')
                first_result['violation_detected'] = len(first_result.get('violations', [])) > 0
                first_result['violation_reason'] = first_result['violations'][0]['reason'] if first_result.get('violations') else ''
                return first_result
        
        return {"error": "Recording failed"}
    
    def record_audio_to_file(self, duration, path):
        """Record audio to WAV file"""
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.6)
                audio = r.record(source, duration=duration)
                with open(path, "wb") as f:
                    f.write(audio.get_wav_data())
            return path
        except:
            return None
    
    def transcribe_audio(self, path):
        """Transcribe audio file to text"""
        r = sr.Recognizer()
        try:
            with sr.AudioFile(path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            return text if text.strip() else "[Could not understand audio]"
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError:
            return "[Speech recognition service unavailable]"
        except:
            return "[Could not understand audio]"
    
    def record_continuous_interview(self, questions_list, duration_per_question, ui_callbacks):
        """
        Record ALL questions continuously - continues even if violations occur
        Captures violation images and stores them for display in results
        """
        
        # ========== PRE-TEST SETUP ==========
        ui_callbacks['status_update']("**🔧 Initializing test environment...**")
        setup_success = self.pre_test_setup_phase(ui_callbacks, timeout=90)
        
        if not setup_success:
            return {"error": "Setup phase failed or timeout"}
        
        # ========== INSTRUCTIONS ==========
        ui_callbacks['countdown_update']("✅ Setup complete! Please read the instructions...")
        ui_callbacks['status_update'](f"""
        **📋 TEST INSTRUCTIONS:**
        - You will answer **{len(questions_list)} questions** continuously
        - Each question has **{duration_per_question} seconds** to answer
        - **Important:** Even if a violation is detected, the interview will continue
        - All violations will be reviewed at the end
        - Stay within boundaries and maintain focus throughout
        
        **The test will begin in 10 seconds...**
        """)
        time.sleep(10)
        
        # ========== START RECORDING ==========
        all_results = []
        
        for i in range(3, 0, -1):
            ui_callbacks['countdown_update'](f"🎬 Test starts in {i}...")
            time.sleep(1)
        ui_callbacks['countdown_update']('')
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "Unable to access camera"}
        
        session_video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        session_video_path = session_video_temp.name
        session_video_temp.close()
        
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(session_video_path, fourcc, 15.0, (640, 480))
        
        session_start_time = time.time()
        session_violations = []
        
        # ========== LOOP THROUGH ALL QUESTIONS ==========
        for q_idx, question_data in enumerate(questions_list):
            
            ui_callbacks['countdown_update'](f"📝 Question {q_idx + 1} of {len(questions_list)}")
            
            question_text = question_data.get('question', 'No question text')
            question_tip = question_data.get('tip', 'Speak clearly and confidently')
            
            ui_callbacks['question_update'](q_idx + 1, question_text, question_tip)
            
            ui_callbacks['status_update'](f"""
            **⏱️ Recording Question {q_idx + 1}**
            
            Time to answer: **{duration_per_question} seconds**
            """)
            
            for i in range(3, 0, -1):
                ui_callbacks['timer_update'](f"⏱️ Starting in {i}s...")
                time.sleep(1)
            
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = audio_temp.name
            audio_temp.close()
            
            audio_thread = threading.Thread(
                target=lambda path=audio_path: self.record_audio_to_file(duration_per_question, path),
                daemon=True
            )
            audio_thread.start()
            
            # Question recording state
            question_start_time = time.time()
            frames = []
            question_violations = []  # Store violations for THIS question
            
            no_face_start = None
            look_away_start = None
            
            eye_contact_frames = 0
            total_frames = 0
            blink_count = 0
            prev_blink = False
            
            face_box = None
            
            # ========== RECORDING LOOP FOR THIS QUESTION ==========
            while (time.time() - question_start_time) < duration_per_question:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames.append(frame.copy())
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                total_frames += 1
                
                lighting_status, brightness = self.analyze_lighting(frame)
                
                num_faces = 0
                looking_at_camera = False
                attention_status = "No Face"
                
                # ========== FACE DETECTION & VIOLATION CHECKS ==========
                if self.models['face_mesh'] is not None:
                    face_results = self.models['face_mesh'].process(rgb_frame)
                    
                    if face_results.multi_face_landmarks:
                        num_faces = len(face_results.multi_face_landmarks)
                        
                        # Check multiple bodies
                        is_multi_body, multi_msg, body_count = self.detect_multiple_bodies(frame, num_faces)
                        
                        if is_multi_body:
                            violation_img_path = self.save_violation_image(frame, q_idx + 1, multi_msg)
                            question_violations.append({
                                'reason': multi_msg,
                                'timestamp': time.time() - question_start_time,
                                'image_path': violation_img_path
                            })
                            # Continue to next question instead of breaking
                            break
                        
                        if num_faces > 1:
                            violation_msg = f"Multiple persons detected ({num_faces} faces)"
                            violation_img_path = self.save_violation_image(frame, q_idx + 1, violation_msg)
                            question_violations.append({
                                'reason': violation_msg,
                                'timestamp': time.time() - question_start_time,
                                'image_path': violation_img_path
                            })
                            break
                        
                        elif num_faces == 1:
                            no_face_start = None
                            face_landmarks = face_results.multi_face_landmarks[0]
                            
                            try:
                                landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                                x_coords = landmarks_2d[:, 0]
                                y_coords = landmarks_2d[:, 1]
                                face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
                                           int(np.max(x_coords) - np.min(x_coords)), 
                                           int(np.max(y_coords) - np.min(y_coords)))
                                
                                # Check boundaries
                                within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, face_box)
                                
                                if not within_bounds:
                                    violation_img_path = self.save_violation_image(frame, q_idx + 1, boundary_msg)
                                    question_violations.append({
                                        'reason': boundary_msg,
                                        'timestamp': time.time() - question_start_time,
                                        'image_path': violation_img_path
                                    })
                                    break
                                
                                # Check person outside frame
                                outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
                                
                                if outside_detected:
                                    violation_msg = f"{obj_type.upper()} detected outside frame ({location} side)"
                                    violation_img_path = self.save_violation_image(frame, q_idx + 1, violation_msg)
                                    question_violations.append({
                                        'reason': violation_msg,
                                        'timestamp': time.time() - question_start_time,
                                        'image_path': violation_img_path
                                    })
                                    break
                                
                                # Check intrusions
                                is_intrusion, intrusion_msg = self.detect_intrusion_at_edges(frame, face_box)
                                if is_intrusion:
                                    violation_img_path = self.save_violation_image(frame, q_idx + 1, intrusion_msg)
                                    question_violations.append({
                                        'reason': intrusion_msg,
                                        'timestamp': time.time() - question_start_time,
                                        'image_path': violation_img_path
                                    })
                                    break
                                
                                # Check hands outside
                                is_hand_violation, hand_msg = self.detect_hands_outside_main_person(frame, face_box)
                                if is_hand_violation:
                                    violation_img_path = self.save_violation_image(frame, q_idx + 1, hand_msg)
                                    question_violations.append({
                                        'reason': hand_msg,
                                        'timestamp': time.time() - question_start_time,
                                        'image_path': violation_img_path
                                    })
                                    break
                                
                                # Suspicious movements
                                is_suspicious, sus_msg = self.detect_suspicious_movements(frame)
                                if is_suspicious:
                                    violation_img_path = self.save_violation_image(frame, q_idx + 1, sus_msg)
                                    question_violations.append({
                                        'reason': sus_msg,
                                        'timestamp': time.time() - question_start_time,
                                        'image_path': violation_img_path
                                    })
                                    break
                                
                                yaw, pitch, roll = self.estimate_head_pose(face_landmarks, frame.shape)
                                gaze_centered = self.calculate_eye_gaze(face_landmarks, frame.shape)
                                
                                is_blink = self.detect_blink(face_landmarks)
                                if is_blink and not prev_blink:
                                    blink_count += 1
                                prev_blink = is_blink
                                
                                head_looking_forward = abs(yaw) <= 20 and abs(pitch) <= 20
                                
                                if head_looking_forward and gaze_centered:
                                    look_away_start = None
                                    looking_at_camera = True
                                    eye_contact_frames += 1
                                    attention_status = "Looking at Camera ✓"
                                else:
                                    if look_away_start is None:
                                        look_away_start = time.time()
                                        attention_status = "Looking Away"
                                    else:
                                        elapsed = time.time() - look_away_start
                                        if elapsed > 2.0:
                                            violation_msg = "Looking away for >2 seconds"
                                            violation_img_path = self.save_violation_image(frame, q_idx + 1, violation_msg)
                                            question_violations.append({
                                                'reason': violation_msg,
                                                'timestamp': time.time() - question_start_time,
                                                'image_path': violation_img_path
                                            })
                                            break
                                        else:
                                            attention_status = f"Looking Away ({elapsed:.1f}s)"
                            except:
                                attention_status = "Face Error"
                    else:
                        if no_face_start is None:
                            no_face_start = time.time()
                            attention_status = "No Face Visible"
                        else:
                            elapsed = time.time() - no_face_start
                            if elapsed > 2.0:
                                violation_msg = "No face visible for >2 seconds"
                                violation_img_path = self.save_violation_image(frame, q_idx + 1, violation_msg)
                                question_violations.append({
                                    'reason': violation_msg,
                                    'timestamp': time.time() - question_start_time,
                                    'image_path': violation_img_path
                                })
                                break
                            else:
                                attention_status = f"No Face ({elapsed:.1f}s)"
                
                # Check for new objects
                if total_frames % 20 == 0:
                    new_detected, new_items = self.detect_new_objects(frame)
                    if new_detected:
                        violation_msg = f"New item(s) brought into view: {', '.join(new_items)}"
                        violation_img_path = self.save_violation_image(frame, q_idx + 1, violation_msg)
                        question_violations.append({
                            'reason': violation_msg,
                            'timestamp': time.time() - question_start_time,
                            'image_path': violation_img_path
                        })
                        break
                
                # Display frame
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
                frame_display = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                
                # Show violation warning if any occurred
                status_color = (0, 255, 0) if len(question_violations) == 0 else (0, 165, 255)
                violation_text = f" | ⚠️ {len(question_violations)} violation(s)" if question_violations else ""
                
                cv2.putText(frame_display, f"Q{q_idx+1}/{len(questions_list)} - {attention_status}{violation_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(frame_display, f"Lighting: {lighting_status}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Eye Contact: {int((eye_contact_frames/max(total_frames,1))*100)}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                elapsed_q = time.time() - question_start_time
                remaining = max(0, int(duration_per_question - elapsed_q))
                cv2.putText(frame_display, f"Time: {remaining}s", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                ui_callbacks['video_update'](cv2.resize(frame_display, (480, 360)))
                
                eye_contact_pct = (eye_contact_frames / max(total_frames, 1)) * 100
                status_text = f"""
                **Question {q_idx + 1} of {len(questions_list)}**
                
                👁️ **Eye Contact:** {eye_contact_pct:.1f}%  
                😴 **Blinks:** {blink_count}  
                💡 **Lighting:** {lighting_status}  
                ⚠️ **Status:** {attention_status}
                """
                
                if question_violations:
                    status_text += f"\n\n⚠️ **Violations in this question:** {len(question_violations)}"
                
                ui_callbacks['status_update'](status_text)
                
                overall_progress = (q_idx + (elapsed_q / duration_per_question)) / len(questions_list)
                overall_progress = max(0.0, min(1.0, overall_progress))
                ui_callbacks['progress_update'](overall_progress)
                ui_callbacks['timer_update'](f"🎥 Q{q_idx+1}/{len(questions_list)} - {remaining}s remaining")
                
                time.sleep(0.05)
            
            # Wait for audio
            audio_thread.join(timeout=duration_per_question + 5)
            
            # Transcribe
            transcript = ""
            if os.path.exists(audio_path):
                transcript = self.transcribe_audio(audio_path)
            
            # Add violations to session list
            if question_violations:
                session_violations.extend([f"Q{q_idx+1}: {v['reason']}" for v in question_violations])
            
            # Store results for this question
            question_result = {
                'question_number': q_idx + 1,
                'question_text': question_data.get('question', ''),
                'audio_path': audio_path,
                'frames': frames,
                'violations': question_violations,  # Now includes image paths
                'violation_detected': len(question_violations) > 0,
                'eye_contact_pct': (eye_contact_frames / max(total_frames, 1)) * 100,
                'blink_count': blink_count,
                'face_box': face_box,
                'transcript': transcript,
                'lighting_status': lighting_status
            }
            
            all_results.append(question_result)
            
            # Show message and continue to next question
            if question_violations:
                ui_callbacks['countdown_update'](f"⚠️ Violation detected in Q{q_idx + 1}! Continuing to next question in 3s...")
                time.sleep(3)
            elif q_idx < len(questions_list) - 1:
                ui_callbacks['countdown_update'](f"✅ Question {q_idx + 1} complete! Next question in 3s...")
                time.sleep(3)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Clear UI
        ui_callbacks['video_update'](None)
        ui_callbacks['progress_update'](1.0)
        
        # Final message
        total_violations = sum(len(r.get('violations', [])) for r in all_results)
        
        if total_violations > 0:
            ui_callbacks['countdown_update'](f"⚠️ TEST COMPLETED WITH {total_violations} VIOLATION(S)")
            ui_callbacks['status_update'](f"**⚠️ {total_violations} violation(s) detected across all questions. Review results below.**")
        else:
            ui_callbacks['countdown_update']("✅ TEST COMPLETED SUCCESSFULLY!")
            ui_callbacks['status_update']("**All questions answered with no violations. Processing results...**")
        
        ui_callbacks['timer_update']("")
        
        # Return comprehensive results
        return {
            'questions_results': all_results,
            'session_video_path': session_video_path,
            'total_questions': len(questions_list),
            'completed_questions': len(all_results),
            'session_violations': session_violations,
            'total_violations': total_violations,
            'violation_images_dir': self.violation_images_dir,
            'session_duration': time.time() - session_start_time
        }

####

