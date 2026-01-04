import cv2
import torch
import json
from model import PoseNet
from torchvision import transforms
from PIL import Image

POSE_NAMES = ["pose1","pose2","pose3","pose4","pose5","pose6"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

num_keypoints = 99  # adjust if different
model = PoseNet(num_keypoints)
model.load_state_dict(torch.load("pose_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame  # replace with mediapipe skeleton frame
    img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(DEVICE)

    with open("current_pose.json") as f:
        data = json.load(f)

    keypoints = []
    for k in sorted(data.keys()):
        keypoints.extend([data[k]["x"], data[k]["y"], data[k]["z"]])

    keypoints = torch.tensor(keypoints).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img, keypoints)
        pred = torch.argmax(out, dim=1).item()

    cv2.putText(frame, POSE_NAMES[pred], (40,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
