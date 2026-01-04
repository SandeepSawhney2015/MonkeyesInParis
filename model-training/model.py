import torch
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self, num_keypoints, num_classes=6):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out = 512

        self.keypoint_fc = nn.Sequential(
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
            nn.Linear(256, num_classes)
        )

    def forward(self, image, keypoints):
        img_feat = self.cnn(image).squeeze(-1).squeeze(-1)
        kp_feat = self.keypoint_fc(keypoints)
        combined = torch.cat((img_feat, kp_feat), dim=1)
        return self.classifier(combined)
