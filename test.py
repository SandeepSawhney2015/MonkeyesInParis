import mediapipe as mp
import tensorflow as tf
import numpy as np

print("=== TEST 1: Import Check ===")
print("TensorFlow version:", tf.__version__)
print("MediaPipe imported successfully!")

print("\n=== TEST 2: TensorFlow Test ===")
# Simple TF operation
x = tf.constant([[5.0, 3.0], [2.0, 7.0]])
y = x * 3
print("Tensor:\n", x.numpy())
print("Tensor * 3:\n", y.numpy())

print("\n=== TEST 3: MediaPipe Hands Inference (No cv2) ===")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Dummy black image, no cv2, just numpy
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Convert to RGB manually (cv2 normally does this)
img_rgb = img[:, :, ::-1]  # reverse color channels

results = hands.process(img_rgb)

print("Inference complete. Detected hands:", results.multi_hand_landmarks)

print("\n=== All tests PASSED ===")
