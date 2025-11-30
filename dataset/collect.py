import cv2
import os

# -----------------------------
# Select pose by keyboard
# -----------------------------
print("Press 1, 2, 3, or 4 to choose pose:")
print("1 = pose1")
print("2 = pose2")
print("3 = pose3")
print("4 = pose4")
print("---------------------------")

pose_key = None
while pose_key not in [49, 50, 51, 52]:  # ASCII codes for 1–4
    pose_key = cv2.waitKey(0)

pose_map = {
    49: "pose1",
    50: "pose2",
    51: "pose3",
    52: "pose4"
}

POSE_NAME = pose_map[pose_key]
print(f"Selected pose: {POSE_NAME}")

# -----------------------------
# Prepare folders (local directory)
# -----------------------------
OUTPUT_DIR = POSE_NAME
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Count existing files to continue numbering
existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
start_index = len(existing_files) + 1

# -----------------------------
# Start camera
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to take a photo")
print("Press Q to quit")

img_index = start_index

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    # SPACE saves a photo
    if key == 32:
        filename = f"{POSE_NAME}_{img_index:03d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")

        img_index += 1

    # Q quits
    elif key == ord('q') or key == ord('Q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
