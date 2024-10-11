# Install required libraries
#!pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def create_human_mesh(landmarks, image_shape):
    height, width = image_shape[:2]
    mesh = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
    
    # Define body landmarks
    left_shoulder = landmarks.pose_landmarks.landmark[11]
    right_shoulder = landmarks.pose_landmarks.landmark[12]
    left_hip = landmarks.pose_landmarks.landmark[23]
    right_hip = landmarks.pose_landmarks.landmark[24]
    
    # Calculate body dimensions
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * width
    hip_width = shoulder_width * 1.1  # Slightly wider hips
    
    # Create smoother body shape
    body_points = np.array([
        [int(left_shoulder.x * width), int(left_shoulder.y * height)],
        [int(right_shoulder.x * width), int(right_shoulder.y * height)],
        [int(right_hip.x * width + hip_width * 0.1), int(right_hip.y * height)],
        [int(left_hip.x * width - hip_width * 0.1), int(left_hip.y * height)]
    ])
    
    # Create a smooth curve for the body
    body_curve = cv2.approxPolyDP(body_points, 0.01 * cv2.arcLength(body_points, True), True)
    cv2.fillPoly(mesh, [body_curve], (200, 150, 150))
    
    # Draw head
    nose = landmarks.pose_landmarks.landmark[0]
    head_radius = int(shoulder_width * 0.4)  # Slightly larger head
    head_center = (int(nose.x * width), int(nose.y * height))
    cv2.circle(mesh, head_center, head_radius, (200, 150, 150), -1)
    
    # Draw a simple face
    draw_simple_face(mesh, head_center, head_radius)
    
    # Draw arms and legs with smooth curves
    limbs = [(11, 13, 15), (12, 14, 16), (23, 25, 27, 31), (24, 26, 28, 32)]
    for limb in limbs:
        points = np.array([(int(landmarks.pose_landmarks.landmark[i].x * width), 
                            int(landmarks.pose_landmarks.landmark[i].y * height)) for i in limb])
        
        # Create a smooth curve for each limb
        curve = cv2.approxPolyDP(points, 0.01 * cv2.arcLength(points, False), False)
        cv2.polylines(mesh, [curve], False, (180, 130, 130), 20, cv2.LINE_AA)
        
        # Draw joints
        for point in points:
            cv2.circle(mesh, tuple(point), 10, (160, 110, 110), -1, cv2.LINE_AA)
    
    return mesh

def draw_simple_face(mesh, center, radius):
    x, y = center
    # Eyes
    eye_y = int(y - radius * 0.15)
    left_eye = (int(x - radius * 0.3), eye_y)
    right_eye = (int(x + radius * 0.3), eye_y)
    cv2.circle(mesh, left_eye, int(radius * 0.12), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(mesh, right_eye, int(radius * 0.12), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(mesh, left_eye, int(radius * 0.06), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(mesh, right_eye, int(radius * 0.06), (0, 0, 0), -1, cv2.LINE_AA)
    
    # Mouth
    mouth_y = int(y + radius * 0.3)
    cv2.ellipse(mesh, (x, mouth_y), (int(radius * 0.3), int(radius * 0.1)), 
                0, 0, 180, (150, 50, 50), -1, cv2.LINE_AA)

def draw_hand(drawing, hand_landmarks, is_left_hand):
    if hand_landmarks:
        # Define colors for each finger (thumb, index, middle, ring, pinky)
        finger_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255)   # Magenta
        ]
        
        # Define finger landmarks
        finger_landmarks = [
            [1, 2, 3, 4],     # Thumb
            [5, 6, 7, 8],     # Index
            [9, 10, 11, 12],  # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20]  # Pinky
        ]
        
        # Draw palm
        palm_landmarks = [0, 1, 5, 9, 13, 17]
        palm_points = [hand_landmarks.landmark[i] for i in palm_landmarks]
        palm_points = np.array([(int(point.x * drawing.shape[1]), int(point.y * drawing.shape[0])) for point in palm_points])
        cv2.fillConvexPoly(drawing, palm_points, (200, 200, 200))
        
        # Draw fingers
        for finger, color in zip(finger_landmarks, finger_colors):
            points = [hand_landmarks.landmark[i] for i in finger]
            points = np.array([(int(point.x * drawing.shape[1]), int(point.y * drawing.shape[0])) for point in points])
            
            # Create a smooth curve for each finger
            curve = cv2.approxPolyDP(points, 0.01 * cv2.arcLength(points, False), False)
            cv2.polylines(drawing, [curve], False, color, 8, cv2.LINE_AA)
            
            # Draw joints
            for point in points:
                cv2.circle(drawing, tuple(point), 5, color, -1, cv2.LINE_AA)

def process_video(input_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output path
    input_filename = os.path.basename(input_path)
    output_filename = f"animated_{input_filename}"
    output_path = os.path.join(output_folder, output_filename)
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Create a black background for each frame
            mesh = np.zeros((height, width, 3), dtype=np.uint8)
            
            if results.pose_landmarks:
                mesh = create_human_mesh(results, frame.shape)
                
                # Draw hands
                draw_hand(mesh, results.left_hand_landmarks, True)   # Left hand
                draw_hand(mesh, results.right_hand_landmarks, False) # Right hand
            
            out.write(mesh)
    
    cap.release()
    out.release()
    print(f"Video processed and saved to: {output_path}")

# Example usage
input_path = 'real_vids/translate.mp4'
output_folder = 'hpe_vids'
process_video(input_path, output_folder)