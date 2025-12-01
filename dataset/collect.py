import cv2
import os
import time
import json
import mediapipe as mp

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
POSES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

DATASET_DIR = "dataset"

# -------------------------------------------
# CREATE DIRECTORIES IF NEEDED
# -------------------------------------------
for pose_name in POSES:
    pose_path = os.path.join(DATASET_DIR, pose_name)
    os.makedirs(pose_path, exist_ok=True)


# -------------------------------------------
# INIT MEDIAPIPE MODELS
# -------------------------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose()
hand_detector = mp_hands.Hands(max_num_hands=2)
face_detector = mp_face.FaceMesh(max_num_faces=1)


# -------------------------------------------
# SELECT POSE
# -------------------------------------------
print()
for i, p in enumerate(POSES, start=1):
    print(f"{i}. {p}")
choice = int(input(f"Enter pose number (1-{len(POSES)}): ").strip())

pose_name = POSES[choice - 1]
pose_dir = os.path.join(DATASET_DIR, pose_name)

print(f"\n>>> Collecting data for: {pose_name}")
print("Press SPACE to capture.\nPress Q to quit.\n")


# -------------------------------------------
# FIND NEXT FILE NUMBER
# -------------------------------------------
existing = [f for f in os.listdir(pose_dir) if f.endswith(".jpg")]
if existing:
    nums = [int(f.split(".")[0]) for f in existing]
    next_num = max(nums) + 1
else:
    next_num = 1


# -------------------------------------------
# OPEN CAMERA
# -------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not access webcam.")
    exit()


# -------------------------------------------
# MAIN LOOP
# -------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose_detector.process(rgb)
    hands_results = hand_detector.process(rgb)
    face_results = face_detector.process(rgb)

    # Show landmarks on screen ONLY
    disp = frame.copy()

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            disp, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_results.multi_hand_landmarks:
        for hand in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                disp, hand, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                disp, face, mp_face.FACEMESH_TESSELATION)

    cv2.putText(disp, f"{pose_name}  Next: {next_num}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting Data", disp)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # SPACEBAR → SAVE 1 FRAME
    # -----------------------------
    if key == 32:  # space bar
        img_path = os.path.join(pose_dir, f"{next_num}.jpg")
        json_path = os.path.join(pose_dir, f"{next_num}.json")

        # Save clean frame
        cv2.imwrite(img_path, frame)

        # Save landmarks in JSON
        data = {
            "pose": [],
            "hands": [],
            "face": []
        }

        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                data["pose"].append([lm.x, lm.y, lm.z])

        if hands_results.multi_hand_landmarks:
            for hand in hands_results.multi_hand_landmarks:
                points = []
                for lm in hand.landmark:
                    points.append([lm.x, lm.y, lm.z])
                data["hands"].append(points)

        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                points = []
                for lm in face.landmark:
                    points.append([lm.x, lm.y, lm.z])
                data["face"].append(points)

        with open(json_path, "w") as f:
            json.dump(data, f)

        print(f"Captured {next_num}.jpg")
        next_num += 1

    # -----------------------------
    # Q → Quit
    # -----------------------------
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("DONE!")
