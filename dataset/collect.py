import cv2
import os
import time
import json
import mediapipe as mp

# ------ CONFIG ------
BURST_SIZE = 300
DELAY_BETWEEN_PHOTOS = 0.2
POSES = ["pose1", "pose2", "pose3", "pose4"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ---------------------------------------------
# Find next image index (avoids overwrites)
# ---------------------------------------------
def get_next_index(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    if not existing:
        return 1

    numbers = []
    for f in existing:
        name = f.replace(".jpg", "")
        try:
            numbers.append(int(name))
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
# Collect images & JSON data for one pose
# ---------------------------------------------
def collect_for_pose(pose):
    folder = os.path.join(os.getcwd(), pose)
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera.")
        return

    print(f"\n📸 Ready to collect for: {pose}")
    print("Press ENTER to begin capturing...")
    input()

    next_index = get_next_index(folder)

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_tracker:

        for i in range(BURST_SIZE):
            success, frame = cap.read()
            if not success:
                print("❌ ERROR: Frame failed.")
                break

            # Process mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)

            # Draw nodes and connections on frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Save landmark JSON
                json_path = os.path.join(folder, f"{next_index + i}.json")
                with open(json_path, "w") as jf:
                    json.dump(landmarks_to_dict(results.pose_landmarks), jf, indent=2)

            # Save the image
            img_path = os.path.join(folder, f"{next_index + i}.jpg")
            cv2.imwrite(img_path, frame)

            # Show the live capturing window
            cv2.imshow("Capturing (ESC to stop)", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("⛔ Stopped early.")
                break

            time.sleep(DELAY_BETWEEN_PHOTOS)

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished capturing {BURST_SIZE} images for '{pose}'")
    print(f"📁 Saved to: {folder}")


# ---------------------------------------------
# PROGRAM START
# ---------------------------------------------
if __name__ == "__main__":
    print("\nSelect the pose to capture:\n")
    for i, pose in enumerate(POSES, start=1):
        print(f"{i}. {pose}")

    choice = int(input("\nEnter number: "))

    if 1 <= choice <= len(POSES):
        collect_for_pose(POSES[choice - 1])
    else:
        print("❌ Invalid selection.")
