# ============================================================
# SILENCE ALL TENSORFLOW / MEDIAPIPE / ABSL WARNINGS
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TF warnings

import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # hide MP logs

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)           # hide Abseil logs
absl_logging.set_stderrthreshold(absl_logging.FATAL)     # block STDERR spam

# ============================================================
# IMPORTS
# ============================================================
import cv2
import mediapipe as mp
import json
from colorama import Fore, Style, init
init()

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
POSE_NAMES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

RAW_SUFFIX = "_raw.jpg"
NODE_SUFFIX = "_nodes.jpg"
MAX_PHOTOS = 300

# MICRO-NODES SETTINGS
SMALL_RADIUS = 1
SMALL_THICKNESS = 1

# -------------------------------------------------------
# MEDIAPIPE SETUP
# -------------------------------------------------------
mp_draw = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Ultra-small landmark specs
small_landmark_style = mp_draw.DrawingSpec(
    color=(0, 255, 0),
    thickness=SMALL_THICKNESS,
    circle_radius=SMALL_RADIUS
)

small_connection_style = mp_draw.DrawingSpec(
    color=(0, 200, 200),
    thickness=SMALL_THICKNESS,
    circle_radius=SMALL_RADIUS
)

holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True
)

# -------------------------------------------------------
# TERMINAL UI
# -------------------------------------------------------
def banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(Fore.CYAN + "═══════════════════════════════════════════════")
    print("            MONKEYSINPARIS DATA COLLECTOR")
    print("═══════════════════════════════════════════════" + Style.RESET_ALL)
    print(Fore.YELLOW + "Press SPACE to capture | ESC to quit" + Style.RESET_ALL)
    print()

def choose_pose():
    print(Fore.GREEN + "Available Poses:" + Style.RESET_ALL)
    for i, p in enumerate(POSE_NAMES, 1):
        print(f"  {i}. {p}")
    print()
    choice = int(input(Fore.CYAN + "Enter pose number (1–6): " + Style.RESET_ALL))
    return POSE_NAMES[choice - 1]

# -------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------
def save_json(filepath, results):
    data = {
        "pose_landmarks": [],
        "face_landmarks": [],
        "left_hand_landmarks": [],
        "right_hand_landmarks": []
    }

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            data["pose_landmarks"].append([lm.x, lm.y, lm.z, lm.visibility])

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            data["face_landmarks"].append([lm.x, lm.y, lm.z])

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            data["left_hand_landmarks"].append([lm.x, lm.y, lm.z])

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            data["right_hand_landmarks"].append([lm.x, lm.y, lm.z])

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------------------------------------
# FILE COUNTER
# -------------------------------------------------------
def get_next_index(folder, pose):
    if not os.path.exists(folder):
        return 1

    files = [f for f in os.listdir(folder) if f.startswith(pose)]
    if not files:
        return 1

    nums = []
    for name in files:
        try:
            num = int(name.split("_")[1].split(".")[0])
            nums.append(num)
        except:
            pass

    return max(nums) + 1 if nums else 1

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
banner()
pose = choose_pose()
banner()
print(Fore.MAGENTA + f">>> Collecting: {pose}" + Style.RESET_ALL)

# Save pose folders in SAME directory as collect.py
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, pose)
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(Fore.RED + "ERROR: Cannot open webcam" + Style.RESET_ALL)
    exit()

index = get_next_index(output_folder, pose)

while True:
    ret, frame = cap.read()
    if not ret:
        print(Fore.RED + "Failed to read frame." + Style.RESET_ALL)
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    raw_frame = frame.copy()
    nodes_frame = frame.copy()

    # ----------------------------
    # SAFE MICRO-NODE DRAWING
    # ----------------------------
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            nodes_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=small_landmark_style,
            connection_drawing_spec=small_connection_style
        )

    if results.face_landmarks:
        mp_draw.draw_landmarks(
            nodes_frame,
            results.face_landmarks,
            mp_face.FACEMESH_TESSELATION,
            landmark_drawing_spec=small_landmark_style,
            connection_drawing_spec=small_connection_style
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            nodes_frame,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=small_landmark_style,
            connection_drawing_spec=small_connection_style
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            nodes_frame,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=small_landmark_style,
            connection_drawing_spec=small_connection_style
        )

    # UI overlay
    cv2.putText(nodes_frame, f"POSE: {pose}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(nodes_frame, f"COUNT: {index-1}/{MAX_PHOTOS}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("MonkeysInParis Collector", nodes_frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture
    if key == ord(' '):
        if index > MAX_PHOTOS:
            print(Fore.GREEN + f"\nReached {MAX_PHOTOS} images. Auto-closing..." + Style.RESET_ALL)
            break

        raw_path = os.path.join(output_folder, f"{pose}_{index:03d}{RAW_SUFFIX}")
        nodes_path = os.path.join(output_folder, f"{pose}_{index:03d}{NODE_SUFFIX}")
        json_path = os.path.join(output_folder, f"{pose}_{index:03d}.json")

        cv2.imwrite(raw_path, raw_frame)
        cv2.imwrite(nodes_path, nodes_frame)
        save_json(json_path, results)

        print(Fore.GREEN + f"Saved {pose}_{index:03d}" + Style.RESET_ALL)
        index += 1

        if index > MAX_PHOTOS:
            print(Fore.GREEN + f"\nReached {MAX_PHOTOS} images. Auto-closing..." + Style.RESET_ALL)
            break

    # ESC = quit
    if key == 27:
        print(Fore.RED + "\nESC pressed. Closing program..." + Style.RESET_ALL)
        break

cap.release()
cv2.destroyAllWindows()
