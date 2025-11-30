import cv2
import os
import time

BURST_COUNT = 300
DELAY_BETWEEN_PHOTOS = 0.2  # seconds


def ensure_pose_folders():
    """Create pose1–pose4 folders if they do not exist."""
    for i in range(1, 5):
        folder = f"pose{i}"
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_next_index(folder):
    """Return next image index for the given pose folder."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]

    numbers = []
    for f in files:
        try:
            num = int(f.split("_")[1].split(".")[0])
            numbers.append(num)
        except:
            pass

    return max(numbers) + 1 if numbers else 1


def main():
    ensure_pose_folders()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Camera not found.")
        return

    current_pose = None
    capturing = False
    photos_taken = 0
    next_index = 1

    print("\nControls:")
    print("  1–4 = Select pose folder")
    print("  Hold SPACE = Capture images")
    print("  Release SPACE = Stop capturing")
    print("  ESC = Quit program\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        cv2.imshow("Capture Window", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == 27:  # ESC
            break

        # Select pose
        if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            pose_number = chr(key)
            current_pose = f"pose{pose_number}"
            next_index = get_next_index(current_pose)
            photos_taken = 0
            capturing = False

            print(f"\n📁 Pose selected: {current_pose}")
            print(f"   Next file index: {next_index}")
            print(f"   Hold SPACE to start capturing...\n")
            continue

        # SPACE pressed → capture starts (only if pose was selected)
        if key == 32:  # SPACE
            if current_pose is not None:
                capturing = True

        # SPACE released
        if key == 255:
            capturing = False

        # Capture logic
        if capturing and current_pose is not None:
            if photos_taken >= BURST_COUNT:
                print("📸 300 photos complete! Press pose number to start new batch.")
                capturing = False
                continue

            filename = f"{current_pose}_{next_index:03d}.jpg"
            filepath = os.path.join(current_pose, filename)
            cv2.imwrite(filepath, frame)

            print(f"Saved: {filepath}")

            next_index += 1
            photos_taken += 1

            time.sleep(DELAY_BETWEEN_PHOTOS)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#FINAL EDITS 11/29/2025
#Authors: Sandeep Sawhney & Ibrahim Quaizar