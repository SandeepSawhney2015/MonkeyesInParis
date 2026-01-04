import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ================= CONFIG =================
DATASET_DIR = "../dataset"
POSES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)

# ================= DATASET =================
class PoseDataset(Dataset):
    def __init__(self, root, split="train"):
        self.samples = []

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for label, pose in enumerate(POSES):
            pose_dir = os.path.join(root, pose)
            files = sorted(os.listdir(pose_dir))

            # find unique prefixes like pose1_0001
            prefixes = sorted(
                set("_".join(f.split("_")[:2]) for f in files if f.endswith(".json"))
            )

            split_idx = int(0.75 * len(prefixes))
            prefixes = prefixes[:split_idx] if split == "train" else prefixes[split_idx:]

            for p in prefixes:
                self.samples.append((
                    os.path.join(pose_dir, f"{p}_nodes.jpg"),
                    os.path.join(pose_dir, f"{p}.json"),
                    label
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        # ---- Image ----
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # ---- JSON (MediaPipe format) ----
        with open(json_path, "r") as f:
            raw = json.load(f)

        pose_landmarks = raw["pose_landmarks"]  # list of [x, y, z, visibility]

        keypoints = []
        for joint in pose_landmarks:
            x, y, z = joint[:3]   # ignore visibility
            keypoints.extend([x, y, z])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        # sanity check (33 joints * 3)
        assert keypoints.shape[0] == 99

        return img, keypoints, label

# ================= MODEL =================
class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out = 512

        self.kp_fc = nn.Sequential(
            nn.Linear(num_keypoints, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(POSES))
        )

    def forward(self, img, kp):
        img_feat = self.cnn(img).squeeze(-1).squeeze(-1)
        kp_feat = self.kp_fc(kp)
        combined = torch.cat((img_feat, kp_feat), dim=1)
        return self.classifier(combined)

# ================= TRAINING =================
def main():
    train_ds = PoseDataset(DATASET_DIR, split="train")
    test_ds = PoseDataset(DATASET_DIR, split="test")

    num_keypoints = 99

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, pin_memory=True
    )

    model = PoseNet(num_keypoints).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, kps, labels in train_loader:
            imgs = imgs.to(DEVICE)
            kps = kps.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs, kps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "pose_model.pth")
    print("Training complete. Model saved as pose_model.pth")

if __name__ == "__main__":
    main()
