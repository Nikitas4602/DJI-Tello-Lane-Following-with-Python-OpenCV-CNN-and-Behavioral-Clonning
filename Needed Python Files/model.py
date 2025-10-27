import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 24, 5, stride=2, padding=2)   # -> 24 x 60 x 80 (από 120x160)
        self.c2 = nn.Conv2d(24, 36, 5, stride=2, padding=2)  # -> 36 x 30 x 40
        self.c3 = nn.Conv2d(36, 48, 5, stride=2, padding=2)  # -> 48 x 15 x 20
        self.c4 = nn.Conv2d(48, 64, 3, stride=1, padding=1)  # -> 64 x 15 x 20
        self.c5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # -> 64 x 15 x 20
        self.dropout = nn.Dropout(p=0.2)

        # LazyLinear: προσαρμόζεται αυτόματα στο flatten μέγεθος (π.χ. 64*15*20=19200)
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)       # [B, C*H*W]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.out(x))   # steer ∈ [-1, 1]
        return x
