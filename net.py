import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def vgg19_bn(num_classes):
    """ImageNet 사전학습 VGG-19-BN을 CIFAR-10 분류용으로 수정한다 (교재 11.9)."""
    # 사전 학습 모델 불러오기(중요1). pretrained=True와 같은 가중치
    net = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)

    # 최종 레이어 함수 교체(중요2): print(net)의 출력 결과로부터 classifier[6]
    last = net.classifier[6]
    assert isinstance(last, nn.Linear)
    net.classifier[6] = nn.Linear(last.in_features, num_classes)

    # features 마지막의(52번째) MaxPool2d 제거(중요3)
    # 중복이라 AdaptiveAvgPool2d가 아무 역할을 못한다. 불필요.
    assert isinstance(net.features, nn.Sequential)
    net.features = net.features[:-1]

    # AdaptiveAvgPool2d 제거(중요4)
    net.avgpool = nn.Identity()  # type: ignore[assignment]
    return net
