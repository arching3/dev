import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 예시 (week10 CNN_v2)
# 모델은 num_classes를 받는 nn.Module 클래스로 정의한다. experiments.py의 Experiment(model=...)에
# 클래스를 그대로 넘기면 train.py가 model(num_classes)로 생성한다. (CIFAR-10 입력: 3x32x32)
#
# class CNN_v2(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                                   # 32 -> 16
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                                   # 16 -> 8
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                                   # 8 -> 4
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(4 * 4 * 128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, num_classes),
#         )
#
#     def forward(self, x):
#         return self.classifier(self.features(x))
