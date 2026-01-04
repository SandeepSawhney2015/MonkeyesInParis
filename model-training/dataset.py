import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

POSE_LABELS = {
    "pose1": 0,
    "pose2": 1,
    "pose3": 2,
    "pose4": 3,
    "pose5": 4,
    "pose6": 5
}

class PoseDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        for pose in POSE_LABELS:
            pose_dir = os.path.join(root_dir, pose)
            files = sorted(os.listdir(pose_dir))

            prefixes = sorted(set(f.split("_")[0] + "_" + f.split("_")[1] for f in files))
            split_idx = int(len(prefixes) * 0.75)

            if split == "train":
                prefixes = prefixes[:split_idx]
            else:
                prefixes = prefixes[split_idx:]

            for p in prefixes:
                self.samples.append((
                    os.path.join(pose_dir, f"{p}_nodes.jpg"),
                    os.path.join(pose_dir, f"{p}.json"),
                    POSE_LABELS[pose]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(json_path) as f:
            data = json.load(f)

        keypoints = []
        for k in sorted(data.keys()):
            keypoints.extend([data[k]["x"], data[k]["y"], data[k]["z"]])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return img, keypoints, label