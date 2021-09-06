import torch.nn as nn


class Expert(nn.Module):

    def __init__(self):
        super(Expert, self).__init__()  # 32x32 --> 32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32))
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()  # 32x32 --> 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2))  # 32x32 --> 16x16
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2))  # 16x16 --> 8x8
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2))  # 8x8 --> 4x4
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.ELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        validity = self.layer1(input)
        validity = self.layer2(validity)
        validity = self.layer3(validity)
        validity = self.layer4(validity)
        validity = self.layer5(validity)
        validity = self.layer6(validity)
        validity = self.layer7(validity)
        validity = validity.view(-1, 64*4*4)
        validity = self.fc1(validity)
        validity = self.fc2(validity)
        return validity