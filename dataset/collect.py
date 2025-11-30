import cv2
import os
import time

# --- CONFIG ---
BURST_SIZE = 300
DELAY_BETWEEN_PHOTOS = 0.2
MAX_PEOPLE = 4
POSES = ["pose1", "pose2", "pose3", "pose4"]

# --- DETERMINE NEXT PERSON BASED ON EXISTING FOLDERS ---
def get_starting_person(pose_name):
    pose_folder = os.path.join("dataset", pose_name)
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

    existing_people = [
        d for d in os.listdir(pose_folder)
        if os.path.isdir(os.path.join(pose_folder, d)) and d.startswith("person")
    ]

    nums = []
    for p in existing_people:
        try:
            nums.append(int(p.replace("person", "")))
        except:
            pass

    if not nums:
        return 1

    highest = max(nums)
    if highest >= MAX_PEOPLE:
        return MAX_PEOPLE
    return highest + 1


# -------------------- MAIN PROGRAM --------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("\n=== Pose Capture Program ===")
print("Press 1–4 to choose a pose.")
print("Press SPACE to start a 300-photo burst for next person.")
print("Press ESC to quit.\n")

selected_pose = None
current_person = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera frame not received.")
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == 27:  # ESC
        print("Exiting program.")
        break

    # Pose selection (1–4)
    if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
        selected_pose = POSES[key - ord("1")]
        current_person = get_starting_person(selected_pose)
        print(f"\nSelected pose: {selected_pose}")
        print(f"Next person: {current_person}")
        continue

    # Start burst on SPACE
    if key == 32:  # SPACE
        if selected_pose is None:
            print("❌ Select a pose first (1–4).")
            continue

        if current_person > MAX_PEOPLE:
            print(f"❌ All {MAX_PEOPLE} people completed for {selected_pose}.")
            print("Select a new pose (1–4).")
            continue

        # Folders
        pose_folder = os.path.join("dataset", selected_pose)
        person_folder = os.path.join(pose_folder, f"person{current_person}")
        os.makedirs(person_folder, exist_ok=True)

        print(f"\n--- Starting burst for {selected_pose}, person {current_person} ---")

        for i in range(1, BURST_SIZE + 1):
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera disconnected.")
                break

            filename = os.path.join(person_folder, f"img{i}.jpg")
            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")
            time.sleep(DELAY_BETWEEN_PHOTOS)

        print(f"\n✔️ Burst complete ({BURST_SIZE} photos).")

        current_person += 1
        if current_person <=_
