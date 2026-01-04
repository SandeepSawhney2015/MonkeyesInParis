import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ================= CONFIG =================
DATASET_DIR = "../dataset"   # folder containing pose1..pose6
POSES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-4
IMG_SIZE = 224

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = (DEVICE == "cuda")
print("Using device:", DEVICE)

# ================= DATASET =================
class PoseDataset(Dataset):
    """
    Expects per pose folder:
      poseX_###_nodes.jpg
      poseX_###.json

    JSON format (MediaPipe Holistic-ish):
      {
        "pose_landmarks": [ [x,y,z,vis], ... 33 joints ... ],
        ...
      }
    """
    def __init__(self, root, split="train"):
        self.samples = []

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for label, pose in enumerate(POSES):
            pose_dir = os.path.join(root, pose)
            if not os.path.isdir(pose_dir):
                raise FileNotFoundError(f"Pose folder not found: {pose_dir}")

            files = os.listdir(pose_dir)

            # --- FIXED PREFIX LOGIC ---
            # prefixes should be like: "pose4_395" (NOT "pose4_395.json")
            prefixes = []
            for f in files:
                if f.endswith(".json"):
                    base = os.path.splitext(f)[0]  # remove .json
                    prefixes.append(base)
            prefixes = sorted(prefixes)

            # 75/25 split per pose
            split_idx = int(0.75 * len(prefixes))
            prefixes = prefixes[:split_idx] if split == "train" else prefixes[split_idx:]

            # Build (img_path, json_path, label)
            for p in prefixes:
                img_path = os.path.join(pose_dir, f"{p}_nodes.jpg")
                json_path = os.path.join(pose_dir, f"{p}.json")
                self.samples.append((img_path, json_path, label))

        # Filter out missing files up front (prevents crashing mid-epoch)
        cleaned = []
        missing = 0
        for img_path, json_path, label in self.samples:
            if os.path.exists(img_path) and os.path.exists(json_path):
                cleaned.append((img_path, json_path, label))
            else:
                missing += 1

        self.samples = cleaned
        if missing > 0:
            print(f"[WARN] Skipped {missing} samples due to missing img/json files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        # Image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # JSON -> pose_landmarks -> 33 joints * (x,y,z) = 99 floats
        with open(json_path, "r") as f:
            raw = json.load(f)

        if "pose_landmarks" not in raw:
            raise ValueError(f"JSON missing 'pose_landmarks': {json_path}")

        pose_landmarks = raw["pose_landmarks"]

        # Some files might be empty / partial; handle gracefully
        if not isinstance(pose_landmarks, list) or len(pose_landmarks) < 33:
            raise ValueError(f"Invalid pose_landmarks in: {json_path}")

        keypoints = []
        for joint in pose_landmarks[:33]:
            # joint is [x,y,z,vis] (or sometimes [x,y,z])
            if not isinstance(joint, list) or len(joint) < 3:
                raise ValueError(f"Bad joint format in: {json_path}")
            x, y, z = joint[0], joint[1], joint[2]
            keypoints.extend([x, y, z])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        # sanity check
        if keypoints.shape[0] != 99:
            raise ValueError(f"Expected 99 keypoint values, got {keypoints.shape[0]} in {json_path}")

        return img, keypoints, label

# ================= MODEL =================
class PoseNet(nn.Module):
    def __init__(self, num_keypoints=99, num_classes=6):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out = 512

        self.kp_fc = nn.Sequential(
            nn.Linear(num_keypoints, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, kp):
        img_feat = self.cnn(img).squeeze(-1).squeeze(-1)
        kp_feat = self.kp_fc(kp)
        x = torch.cat((img_feat, kp_feat), dim=1)
        return self.classifier(x)

# ================= EVAL =================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for imgs, kps, labels in loader:
        imgs = imgs.to(DEVICE)
        kps = kps.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs, kps)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)

# ================= TRAINING =================
def main():
    train_ds = PoseDataset(DATASET_DIR, split="train")
    test_ds  = PoseDataset(DATASET_DIR, split="test")

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # Windows stability: num_workers=0 avoids multiprocessing hiccups
    pin = (DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    model = PoseNet(num_keypoints=99, num_classes=len(POSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0

        for batch in train_loader:
            # If any sample raises in __getitem__, DataLoader can crash.
            # We filtered missing files up front, but if JSON is malformed,
            # it will raise. If you hit that, we can add a "safe collate_fn".
            imgs, kps, labels = batch

            imgs = imgs.to(DEVICE)
            kps = kps.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs, kps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        avg_loss = total_loss / max(seen, 1)
        acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "pose_model_best.pth")

    torch.save(model.state_dict(), "pose_model_last.pth")
    print(f"Done. Best Acc: {best_acc*100:.2f}%")
    print("Saved: pose_model_best.pth and pose_model_last.pth")

if __name__ == "__main__":
    main()
