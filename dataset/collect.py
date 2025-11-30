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


def get_next_index(folder, pose_name):
    files = [f for f in os.listdir(folder) if f.startswith(pose_name) and f.endswith(".jpg")]
    if not files:
        return 1

    nums = []
    for f in files:
        try:
            num = int(f.replace(pose_name + "_", "").replace(".jpg", ""))
            nums.append(num)
        except:
            pass

    return max(nums) + 1 if nums else 1


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


def collect_for_pose(pose):
    folder = os.path.join(os.getcwd(), pose)
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not access camera.")
        return

    print(f"\n📸 Ready for: {pose}")
    print("👉 HOLD SPACE to capture images")
    print("👉 RELEASE SPACE to pause")
    print("👉 Press ESC to quit anytime\n")

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)

            # CLEAN SAVE FRAME (skeleton only)
            save_frame = frame.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    save_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # DISPLAY FRAME FOR UI (copy of save_frame)
            display_frame = save_frame.copy()

            # ---------- SMALL UI, BOTTOM LEFT ----------
            h, w = display_frame.shape[:2]
            x, y = 20, h - 20

            # Show pose name on preview ONLY (not saved)
            cv2.putText(display_frame, f"Pose: {pose}", (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(display_frame, "Hold SPACE to capture", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(display_frame, f"{captured}/{BURST_SIZE}", (x, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            # ------------------------------------------

            cv2.imshow("Capture (hold SPACE)", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("⛔ Stopped by user.")
                break

            if key == 32:  # SPACE held down
                img_name = f"{pose}_{next_index:03}.jpg"
                json_name = f"{pose}_{next_index:03}.json"

                # SAVE CLEAN SKELETON FRAME
                cv2.imwrite(os.path.join(folder, img_name), save_frame)

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
# PROGRAM START (CLEAR MENU)
# ---------------------------------------------
if __name__ == "__main__":
    print("\n==============================")
    print("      POSE COLLECTION MENU     ")
    print("==============================\n")

    print("Select the pose to capture:\n")

    for i, pose in enumerate(POSES, start=1):
        print(f"{i}. {pose}")

    print("\n---------------------------------------")
    print("👉  ENTER THE NUMBER OF YOUR POSE (e.g., 1)")
    print("---------------------------------------\n")

    choice = input("Enter number here: ")

    if not choice.isdigit():
        print("❌ Invalid input. Please enter a number like 1 or 2.")
        exit()

    choice = int(choice)

    if 1 <= choice <= len(POSES):
        collect_for_pose(POSES[choice - 1])
    else:
        print("❌ Invalid selection.")
#Sandeep Sawhney @ Ibrahim Quaizar
#11/30/2025
