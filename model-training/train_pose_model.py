import os
import json
import random
import time
from pathlib import Path

print("Hello World")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# =========================
# CONFIG (you can edit these)
# =========================
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
IMG_SIZE = 224
SEED = 42

# If you want to FORCE GPU only, set True
REQUIRE_CUDA = False

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = (DEVICE == "cuda")
print("Using device:", DEVICE)
if REQUIRE_CUDA and DEVICE != "cuda":
    raise RuntimeError("CUDA is not available. Fix PyTorch CUDA, or set REQUIRE_CUDA=False.")

# =========================
# DATASET AUTO-DISCOVERY
# =========================
def find_dataset_dir() -> Path:
    """
    Tries common locations relative to this script:
      model-training/train_pose_model.py
      ../dataset
      ../Dataset
      ../data
      ../Data
      ./dataset
      ./data
      and finally asks user to edit a single constant if not found.
    """
    here = Path(__file__).resolve().parent

    candidates = [
        here / ".." / "dataset",
        here / ".." / "Dataset",
        here / ".." / "data",
        here / ".." / "Data",
        here / "dataset",
        here / "data",
        here / ".." / ".." / "dataset",
        here / ".." / ".." / "data",
    ]

    for c in candidates:
        c = c.resolve()
        if c.exists() and c.is_dir():
            # must contain at least one subfolder with jsons
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if any(list(d.glob("*.json")) for d in subdirs):
                return c

    # If none found, raise with helpful info
    msg = (
        "Could not auto-find your dataset directory.\n\n"
        "Fix options:\n"
        "1) Put your dataset folder next to your project like:\n"
        "   MonkeyesInParis/\n"
        "     dataset/\n"
        "     model-training/\n\n"
        "2) Or edit DATASET_DIR_OVERRIDE in this file to the full path.\n"
    )
    raise FileNotFoundError(msg)

# If auto-discovery fails, you can override with a full path like:
# DATASET_DIR_OVERRIDE = r"C:\Users\ssand\MonkeyesInParis\dataset"
DATASET_DIR_OVERRIDE = None

def get_dataset_dir() -> Path:
    if DATASET_DIR_OVERRIDE:
        p = Path(DATASET_DIR_OVERRIDE).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"DATASET_DIR_OVERRIDE path does not exist: {p}")
        return p
    return find_dataset_dir()

def discover_pose_folders(dataset_dir: Path):
    """
    Auto-detect pose folders.
    We define a pose folder as any subdirectory containing .json files.
    """
    pose_dirs = []
    for d in dataset_dir.iterdir():
        if d.is_dir():
            if any(d.glob("*.json")):
                pose_dirs.append(d)
    pose_dirs.sort(key=lambda x: x.name.lower())
    if len(pose_dirs) == 0:
        raise FileNotFoundError(f"No pose folders found inside: {dataset_dir}")
    return pose_dirs

# =========================
# DATASET
# =========================
class PoseDataset(Dataset):
    """
    Per pose folder expects:
      <prefix>_nodes.jpg
      <prefix>.json

    JSON format:
      { "pose_landmarks": [ [x,y,z,vis], ... ] }
    """
    def __init__(self, dataset_dir: Path, pose_dirs, split="train"):
        self.samples = []

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        for label, pose_dir in enumerate(pose_dirs):
            files = list(pose_dir.iterdir())

            prefixes = []
            for f in files:
                if f.suffix.lower() == ".json":
                    prefixes.append(f.stem)  # removes .json
            prefixes.sort()

            split_idx = int(0.75 * len(prefixes))
            prefixes = prefixes[:split_idx] if split == "train" else prefixes[split_idx:]

            for p in prefixes:
                img_path = pose_dir / f"{p}_nodes.jpg"
                json_path = pose_dir / f"{p}.json"
                self.samples.append((img_path, json_path, label))

        # filter missing upfront
        cleaned = []
        missing = 0
        for img_path, json_path, label in self.samples:
            if img_path.exists() and json_path.exists():
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

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(json_path, "r") as f:
            raw = json.load(f)

        pose_landmarks = raw.get("pose_landmarks", None)
        if pose_landmarks is None or not isinstance(pose_landmarks, list) or len(pose_landmarks) < 33:
            raise ValueError(f"Bad pose_landmarks in {json_path}")

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

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, kps, labels = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(kps, 0), torch.tensor(labels, dtype=torch.long)

# =========================
# MODEL
# =========================
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

# =========================
# TRAIN
# =========================
def main():
    dataset_dir = get_dataset_dir()
    pose_dirs = discover_pose_folders(dataset_dir)
    pose_names = [d.name for d in pose_dirs]

    print("Dataset dir:", str(dataset_dir))
    print("Detected pose folders:", pose_names)

    train_ds = PoseDataset(dataset_dir, pose_dirs, split="train")
    test_ds  = PoseDataset(dataset_dir, pose_dirs, split="test")

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    pin = (DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin, collate_fn=safe_collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=pin, collate_fn=safe_collate)

    model = PoseNet(num_keypoints=99, num_classes=len(pose_dirs)).to(DEVICE)
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
