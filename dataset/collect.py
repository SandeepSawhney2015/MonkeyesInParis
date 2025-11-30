import cv2
import os
import time

# -----------------------------------------
# CHANGE THESE FOR EACH PERSON + POSE
pose_name = "pose1"      # e.g. "pose1", "pose2", "pose3", "pose4"
person = "person1"       # e.g. "person1", "person2", "person3", "person4"
max_photos = 300         # how many images per session
delay = 0.2              # seconds between saves (0.2s = 5 photos/sec)
# -----------------------------------------

save_dir = f"dataset/{pose_name}/{person}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera. Run this on your laptop, not in Codespaces.")
    exit()

# Start from however many files already exist
count = len(os.listdir(save_dir))
print(f"[INFO] Starting at image index {count}")
print(f"[INFO] Capturing up to {max_photos} images for {pose_name} ({person})")
print("[INFO] Press ESC to stop early.")

last_time = time.time()

while True:
    if count >= max_photos:
        print(f"[DONE] Reached {max_photos} images for {pose_name} ({person}).")
        break

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from camera.")
        break

    cv2.imshow("Auto Capture", frame)
    key = cv2.waitKey(1)

    # ESC to quit early
    if key == 27:  # ESC
        print("[INFO] Stopped early by user.")
        break

    # Save a frame every `delay` seconds
    if time.time() - last_time >= delay:
        img_path = os.path.join(save_dir, f"{pose_name}_{person}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print("[SAVED]", img_path)
        count += 1
        last_time = time.time()

cap.release()
cv2.destroyAllWindows()
