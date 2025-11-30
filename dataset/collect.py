import cv2
import os
import time

# Settings
BURST_SIZE = 300
DELAY_BETWEEN_SHOTS = 0.2  # seconds

# Create folder structure
def ensure_dirs():
    for pose in range(1, 5):
        pose_dir = f"pose{pose}"
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)
        for person in range(1, 10):  # allow up to 9 people if needed
            person_dir = os.path.join(pose_dir, f"person{person}")
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

ensure_dirs()

def get_next_image_number(folder):
    existing = [int(f.split(".")[0]) for f in os.listdir(folder) if f.endswith(".jpg")]
    return max(existing) + 1 if existing else 1

def run_camera():
    cap = cv2.VideoCapture(0)

    current_pose = None
    person_number = 1

    print("\n=== CONTROLS ===")
    print("Press 1–4 to select pose.")
    print("Press SPACE to start a 300-image burst.")
    print("Press ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Display window
        cv2.putText(frame, f"Pose: {current_pose if current_pose else '-'} | Person: {person_number}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Dataset Collector", frame)

        key = cv2.waitKey(1)

        # ======================
        # POSE SELECTION (1–4)
        # ======================
        if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            current_pose = int(chr(key))
            print(f"\nSelected Pose {current_pose}")
            print("Press SPACE to start capturing...")
            time.sleep(0.2)

        # Quit with ESC
        if key == 27:
            print("Exiting...")
            break

        # No pose selected yet
        if current_pose is None:
            continue

        # ======================
        # SPACE → START BURST
        # ======================
        if key == 32:  # space bar
            pose_folder = f"pose{current_pose}"
            person_folder = os.path.join(pose_folder, f"person{person_number}")

            print(f"\nStarting burst of {BURST_SIZE} photos for Pose {current_pose}, Person {person_number}")

            img_number = get_next_image_number(person_folder)

            for i in range(BURST_SIZE):
                ret, frame = cap.read()
                if not ret:
                    continue

                filename = os.path.join(person_folder, f"{img_number}.jpg")
                cv2.imwrite(filename, frame)
                img_number += 1

                cv2.putText(frame, f"Capturing {i+1}/{BURST_SIZE}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Dataset Collector", frame)

                if cv2.waitKey(1) == 27:  # ESC during burst
                    print("Burst interrupted by ESC. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                time.sleep(DELAY_BETWEEN_SHOTS)

            print(f"Burst complete! {BURST_SIZE} images saved.")
            print("Press SPACE again for another 300, or ESC to exit.\n")

    cap.release()
    cv2.destroyAllWindows()

run_camera()
