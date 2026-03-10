"""
Eye Detection Script using MediaPipe Face Landmarker

This script performs real-time eye detection and blink detection using computer vision.
It captures video from the webcam, detects facial landmarks with MediaPipe, calculates
the Eye Aspect Ratio (EAR) for both eyes, and determines if the eyes are open or closed
based on a threshold. The results are displayed on the video feed with visual overlays.

Dependencies:
- OpenCV (cv2)
- MediaPipe (mediapipe)
- NumPy (numpy)

Usage:
Run the script: python eye_detection.py
Press 'q' to quit the application.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import mediapipe as mp
import numpy as np

def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.

    The EAR is computed using the distances between specific eye landmarks.
    It helps determine if the eye is open or closed.

    Args:
        eye (numpy.ndarray): Array of 6 (x, y) coordinates representing eye landmarks.

    Returns:
        float: The computed Eye Aspect Ratio.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

face_landmarker = FaceLandmarker.create_from_options(options)

# Indices for left and right eye landmarks based on MediaPipe's 478 facial landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Threshold for eye aspect ratio to determine if eyes are closed
EAR_THRESHOLD = 0.25

# Capture video from webcam (device 0)
cap = cv2.VideoCapture(0)

frame_timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process the frame with MediaPipe Face Landmarker for video mode
    results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33  # Approximate timestamp increment for 30 FPS

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # Extract eye landmarks and scale to frame dimensions
            left_eye = np.array([(face_landmarks[i].x * frame.shape[1], face_landmarks[i].y * frame.shape[0]) for i in LEFT_EYE])
            right_eye = np.array([(face_landmarks[i].x * frame.shape[1], face_landmarks[i].y * frame.shape[0]) for i in RIGHT_EYE])

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Determine if eyes are closed based on EAR threshold
            if ear < EAR_THRESHOLD:
                status = "Eyes Closed"
                color = (0, 0, 255)  # Red
            else:
                status = "Eyes Open"
                color = (0, 255, 0)  # Green

            # Print EAR and status for debugging/testing
            print(f"EAR: {ear:.2f}, Status: {status}")

            # Draw the eye landmarks and a bounding polygon on the frame
            # Note: cv2.polylines expects integer coordinates
            left_pts = left_eye.astype(int)
            right_pts = right_eye.astype(int)
            cv2.polylines(frame, [left_pts], isClosed=True, color=color, thickness=1)
            cv2.polylines(frame, [right_pts], isClosed=True, color=color, thickness=1)
            for (x, y) in np.vstack((left_pts, right_pts)):
                cv2.circle(frame, (x, y), 2, color, -1)

            # Display the status on the frame
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame in a window
    cv2.imshow('Eye Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
