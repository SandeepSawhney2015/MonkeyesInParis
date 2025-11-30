import cv2
import os
import time
import json
import mediapipe as mp

# ------ CONFIG ------
BURST_SIZE = 300
DELAY_BETWEEN_PHOTOS = 0.2   # <-- UPDATED TO 0.2s
POSES = ["pose1", "pose2", "pose3", "pose4"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ---------------------------------------------
# Find next available index for pose file naming
# ---------------------------------------------
def get_next_index(folder, pose_name):
    files = [f for f in os.listdir(folder) if f.startswith(pose_name) and f.endswith(".jpg")]
    if not files:
        return 1

    numbers = []
    for f in files:
        try:
            num = int(f.replace(pose_name + "_", "").replace(".jpg", ""))
            numbers.append(num)
        except:
            pass

    return max(numbers) + 1 if numbers else 1


# ---------------------------------------------
# Convert Mediapipe landmarks → Python list
# ---------------------------------------------
def landmarks_to_dict(landmarks):
    return [
        {
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        }
        for lm in landmarks.landmark
    ]


# ---------------------------------------------
# Collect images & JSON for one pose
# ---------------------------------------------
def collect_for_pose(pose):
    folder = os.path.join(os.getcwd(), pose)
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not access camera.")
        return

    print(f"\n📸 Ready for: {pose}")
    print("HOLD SPACE to capture images")
    print("RELEASE SPACE to pause")
    print("Press ESC to quit anytime")

    next_index = get_next_index(folder, pose)
    captured = 0

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_tracker:

        while captured < BURST_SIZE:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)

            # Draw pose nodes on preview
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            cv2.putText(frame, "Hold SPACE to capture", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capture (hold SPACE)", frame)
            key = cv2.waitKey(1) & 0xFF

            # ESC → exit
            if key == 27:
                print("⛔ Stopped by user.")
                break

            # SPACE HELD → save frame + json
            if key == 32:
                img_name = f"{pose}_{next_index:03}.jpg"
                json_name = f"{pose}_{next_index:03}.json"

                cv2.imwrite(os.path.join(folder, img_name), frame)

                if results.pose_landmarks:
                    with open(os.path.join(folder, json_name), "w") as jf:
                        json.dump(landmarks_to_dict(results.pose_landmarks), jf, indent=2)

                print(f"Saved: {img_name}")
                next_index += 1
                captured += 1

                time.sleep(DELAY_BETWEEN_PHOTOS)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished collecting for {pose}")
    print(f"📁 Saved in: {folder}")


# ---------------------------------------------
# PROGRAM START
# ---------------------------------------------
if __name__ == "__main__":
    p
