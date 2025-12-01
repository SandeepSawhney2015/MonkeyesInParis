import cv2
import mediapipe as mp
import os
import time
import json

# ---------------- CONFIG ----------------
BURST_SIZE = 300
DELAY_BETWEEN_PHOTOS = 0.15
SAVE_FOLDER = "dataset"

POSE_NAMES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

# Ask user which pose to collect
print("Which pose do you want to collect?")
for i, name in enumerate(POSE_NAMES, 1):
    print(f"{i}. {name}")

choice = int(input("Enter pose number (1-6): ").strip())
POSE_NAME = POSE_NAMES[choice - 1]

print(f"\n>>> Collecting data for: {POSE_NAME}\n")

# Create folders
pose_folder = os.path.join(SAVE_FOLDER, POSE_NAME)
os.makedirs(pose_folder, exist_ok=True)

# ------------ MEDIAPIPE MODELS ------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,     # gives detailed iris + lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------- CAMERA -----------------
cap = cv2.VideoC
