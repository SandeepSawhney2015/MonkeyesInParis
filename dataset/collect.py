import cv2
import os

def get_next_index(folder, prefix):
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".jpg")]
    if not files:
        return 1
    nums = [int(f.replace(prefix + "_", "").replace(".jpg", "")) for f in files]
    return max(nums) + 1

print("Press 1, 2, 3, or 4 to choose a pose folder.")
print("Then press SPACE to take a photo.")
print("Press Q to quit.")

# Wait for pose selection
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

# Create folder
os.makedirs(selected_pose, exist_ok=True)

# Determine next index
index = get_next_index(selected_pose, selected_pose)

# ---------------------------
# IMPORTANT: Windows camera fix
# ---------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started. Press SPACE to capture, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE saves
        filename = f"{selected_pose}/{selected_pose}_{index:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        index += 1

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
