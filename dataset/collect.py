import cv2
import os
import time

def get_next_index(folder, prefix):
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".jpg")]
    if not files:
        return 1
    nums = [int(f.replace(prefix + "_", "").replace(".jpg", "")) for f in files]
    return max(nums) + 1

print("Press 1–4 to choose a pose folder.")
print("Hold SPACE to take continuous photos.")
print("Press Q to quit.")

# Pose selection
selected_pose = None
while selected_pose is None:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        selected_pose = "pose1"
    elif key == ord('2'):
        selected_pose = "pose2"
    elif key == ord('3'):
        selected_pose = "pose3"
    elif key == ord('4'):
        selected_pose = "pose4"

print(f"Selected: {selected_pose}")

# Folder + numbering
os.makedirs(selected_pose, exist_ok=True)
index = get_next_index(selected_pose, selected_pose)

# Force Windows camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera running. Hold SPACE to capture photos.")

last_capture_time = 0
capture_delay = 0.2  # seconds between captures while holding space

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        print("Exiting...")
        break

    # Hold SPACE to continuously capture
    if key == ord(' '):
        now = time.time()
        if now - last_capture_time >= capture_delay:
            filename = f"{selected_pose}/{selected_pose}_{index:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            index += 1
            last_capture_time = now

cap.release()
cv2.destroyAllWindows()
