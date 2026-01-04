import os
import json
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ================= CONFIG =================
DATASET_DIR = "../dataset"
POSES = ["pose1", "pose2", "pose3", "pose4", "pose5", "pose6"]

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
IMG_SIZE = 224

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = (DEVICE == "cuda")

# If you want to FORCE GPU only, set to True
REQUIRE_CUDA = False

print("Using device:", DEVICE)
if REQUIRE_CUDA and DEVICE != "cuda":
    raise RuntimeError("CUDA is not available. Fix your PyTorch CUDA install, or set REQUIRE_CUDA=False.")

# ================= DATASET =================
class PoseDataset(Dataset):
    def __init__(self, root, split="train"):
        self.samples = []

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        for label, pose in enumerate(POSES):
            pose_dir = os.path.join(root, pose)
            if not os.path.isdir(pose_dir):
                raise FileNotFoundError(f"Pose folder not found: {pose_dir}")

            files = os.listdir(pose_dir)

            # prefixes from json files, stripped correctly
            prefixes = []
            for f in files:
                if f.endswith(".json"):
                    prefixes.append(os.path.splitext(f)[0])  # pose4_395 (no .json)
            prefixes = sorted(prefixes)

            split_idx = int(0.75 * len(prefixes))
            prefixes = prefixes[:split_idx] if split == "train" else prefixes[split_idx:]

            for p in prefixes:
                img_path = os.path.join(pose_dir, f"{p}_nodes.jpg")
                json_path = os.path.join(pose_dir, f"{p}.json")
                self.samples.append((img_path, json_path, label))

        # filter missing files up front
        cleaned = []
        missing = 0
        for img_path, json_path, label in self.samples:
            if os.path.exists(img_path) and os.path.exists(json_path):
                cleaned.append((img_path, json_path, label))
            else:
                missing += 1
        self.samples = cleaned

        if missing:
            print(f"[WARN] Skipped {missing} samples due to missing files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        # image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # json
        with open(json_path, "r") as f:
            raw = json.load(f)

        # must exist
        pose_landmarks = raw.get("pose_landmarks", None)
        if pose_landmarks is None or not isinstance(pose_landmarks, list) or len(pose_landmarks) < 33:
            raise ValueError(f"Bad pose_landmarks in {json_path}")

        # 33 joints * (x,y,z) = 99
        keypoints = []
        for joint in pose_landmarks[:33]:
            if not isinstance(joint, list) or len(joint) < 3:
                raise ValueError(f"Bad joint format in {json_path}")
            x, y, z = joint[0], joint[1], joint[2]
            keypoints.extend([x, y, z])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        if keypoints.numel() != 99:
            raise ValueError(f"Expected 99 values, got {keypoints.numel()} in {json_path}")

        return img, keypoints, label

# ================= SAFE COLLATE =================
def safe_collate(batch):
    """
    Skips any samples that failed in __getitem__.
    Prevents training from "hanging" or crashing on one bad json/image.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, kps, labels = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(kps, 0), torch.tensor(labels, dtype=torch.long)

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

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        if batch is None:
            continue
        imgs, kps, labels = batch
        imgs, kps, labels = imgs.to(DEVICE), kps.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs, kps)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)

# ================= TRAIN =================
def main():
    train_ds = PoseDataset(DATASET_DIR, split="train")
    test_ds  = PoseDataset(DATASET_DIR, split="test")

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    pin = (DEVICE == "cuda")

    # Windows: keep num_workers=0 to avoid multiprocessing weirdness
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin, collate_fn=safe_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin, collate_fn=safe_collate
    )

    model = PoseNet(num_keypoints=99, num_classes=len(POSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        seen = 0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
            imgs, kps, labels = batch
            imgs, kps, labels = imgs.to(DEVICE), kps.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs, kps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

            # progress print every 50 batches
            if (i + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}] batch {i+1} | loss {loss.item():.4f}")

        avg_loss = epoch_loss / max(seen, 1)
        acc = evaluate(model, test_loader)
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}% | Time: {dt:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "pose_model_best.pth")

    torch.save(model.state_dict(), "pose_model_last.pth")
    print(f"Done. Best Acc: {best_acc*100:.2f}%")
    print("Saved: pose_model_best.pth and pose_model_last.pth")

if __name__ == "__main__":
    main()
