import cv2
import os
import time

# Directory to save photos
SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# Webcam
cap = cv2.VideoCapture(0)

current_pose = None        # Pose number (1–4)
burst_active = False       # Whether we're currently capturing
burst_number = 1           # Burst index per pose
photo_count = 0            # Count within current burst

BURST_SIZE = 300           # Number of photos per burst
DELAY = 0.2                # Seconds between photos

print("=================================================")
print("Controls:")
print("   1 = Select Pose 1")
print("   2 = Select Pose 2")
print("   3 = Select Pose 3")
print("   4 = Select Pose 4")
print("   SPACE = Start/Stop a burst of 300 photos")
print("   Q = Quit")
print("=================================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display pose info on video
    display_text = f"Pose: {current_pose if current_pose else 'None'}"
    if burst_active:
        display_text += f" | Capturing {photo_count}/{BURST_SIZE}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # --- Pose selection ---
    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_pose = int(chr(key))
        burst_number = 1  # reset burst count for the pose
        print(f"\n>>> Selected Pose {current_pose}")

    # --- Start/Stop burst ---
    if key == ord(" "):
        if current_pose is None:
            print("!!! Select a pose first (1–4).")
            continue

        burst_active = not burst_active
        if burst_active:
            print(f"\n>>> Starting burst {burst_number} for Pose {current_pose}...")
            photo_count = 0
        else:
            print(f">>> Stopping burst {burst_number} for Pose {current_pose}.")
            burst_number += 1

    # Quit
    if key == ord("q"):
        print("\nExiting...")
        break

    # --- Capture photos ---
    if burst_active and photo_count < BURST_SIZE:
        filename = (
            f"pose_{current_pose}_burst_{burst_number}_img_{photo_count:03d}.jpg"
        )
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        photo_count += 1

        # Delay of 0.2 seconds between captures
        time.sleep(DELAY)

        if photo_count == BURST_SIZE:
            print(f">>> Burst {burst_number} for Pose {current_pose} complete ({BURST_SIZE} photos).")
            burst_active = False
            burst_number += 1

cap.release()
cv2.destroyAllWindows()
