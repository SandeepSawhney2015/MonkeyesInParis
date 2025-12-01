import cv2
import mediapipe as mp
import os
import time
import json

# ---------------- CONFIG ----------------
BURST_SIZE = 300
DELAY_BETWEEN_PHOTOS = 0.15
SAVE_FOLDER = "dataset"

POSE_NAMES = ["pose1", "pose2", "pose3", "pose4", "pose5"]

# Ask for which pose to collect
print("Which pose do you want to collect?")
for i, name in enumerate(POSE_NAMES, 1):
    print(f"{i}. {name}")

choice = int(input("Enter pose number (1-5): ").strip())
POSE_NAME = POSE_NAMES[choice - 1]

print(f"\n>>> Collecting data for: {POSE_NAME}\n")

# Create pose folder
pose_folder = os.path.join(SAVE_FOLDER, POSE_NAME)
os.makedirs(pose_folder, exist_ok=True)

# ------------ MEDIAPIPE MODELS ------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
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

# ------------- CAMERA -----------------
cap = cv2.VideoCapture(0)
time.sleep(1)

# ---------- DETERMINE NEXT IMAGE INDEX ----------
existing = [int(f.split(".")[0]) for f in os.listdir(pose_folder) if f.endswith(".jpg")]
start_index = max(existing) + 1 if existing else 1

print(f"Starting at image index: {start_index}")

# -------------- MAIN LOOP ---------------
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    debug_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run BOTH models
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    # ------- DRAW FINGER LANDMARKS -------
    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(debug_frame, hand, mp_hands.HAND_CONNECTIONS)

    # ------- DRAW POSE LANDMARKS (OPTIONAL) -------
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            debug_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Collecting", debug_frame)
    key = cv2.waitKey(1)

    # SPACE = start burst
    if key == 32:
        print("Capturing burst...")
        for i in range(BURST_SIZE):
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)

            # ---- Extract Landmark Data ----
            data = {"hands": [], "pose": []}

            # Hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    points = []
                    for lm in hand.landmark:
                        points.append([lm.x, lm.y, lm.z])
                    data["hands"].append(points)

            # Pose landmarks
            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks.landmark:
                    data["pose"].append([lm.x, lm.y, lm.z])

            # Save image + JSON
            img_path = os.path.join(pose_folder, f"{start_index}.jpg")
            json_path = os.path.join(pose_folder, f"{start_index}.json")

            cv2.imwrite(img_path, frame)

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved {img_path}")
            start_index += 1
            count += 1
            time.sleep(DELAY_BETWEEN_PHOTOS)

        print("Burst complete.")

    # ESC = exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
