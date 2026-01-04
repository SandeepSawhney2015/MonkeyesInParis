import torch
from torch.utils.data import DataLoader
from dataset import PoseDataset
from model import PoseNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
BATCH_SIZE = 32

train_ds = PoseDataset("../dataset", split="train")
test_ds = PoseDataset("../dataset", split="test")

num_keypoints = len(train_ds[0][1])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

model = PoseNet(num_keypoints).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, kps, labels in train_loader:
        imgs, kps, labels = imgs.to(DEVICE), kps.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs, kps)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "pose_model.pth")
print("Model saved")
