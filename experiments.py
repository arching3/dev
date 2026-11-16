"""실험 정의 레지스트리.

새 실험은 파일 하단에 register(Experiment(...))를 하나 추가하면 된다.
공통 값(optimizer 포함)은 config.py의 Config를 따르고, 실험마다 다른 값만 overrides에 적는다.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping

import torch.nn as nn
from torchvision import transforms

import net


# ---------------------------------------------------------------- transforms
def _resize(size):
    """size가 주어지면 32×32 이미지를 size×size로 키우는 Resize를 맨 앞에 둔다."""
    return [transforms.Resize(size)] if size else []


def base_transform(size=None):
    """증강 없이 (Resize) + 텐서 변환 + 정규화만 적용한다."""
    return transforms.Compose([
        *_resize(size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])


def augment_transform(size=None):
    """(Resize) + RandomHorizontalFlip + RandomErasing 증강."""
    return transforms.Compose([
        *_resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3),
                                 value=0, inplace=False),
    ])


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class Experiment:
    number: int
    key: str                                   # 출력 폴더 이름
    title: str                                 # 로그·그래프 제목
    model: Callable[[int], nn.Module]          # num_classes -> 모델
    train_transform: Callable[[], Callable] = base_transform
    test_transform: Callable[[], Callable] = base_transform
    criterion: Callable[[], nn.Module] = nn.CrossEntropyLoss
    # Config 필드 덮어쓰기. optimizer도 여기서 지정:
    #   {"optimizer": "sgd", "optimizer_params": {"momentum": 0.9}, "lr": 0.01}
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def config(self, base):
        return base.updated(**self.overrides)


EXPERIMENTS: dict[int, Experiment] = {}


def register(experiment):
    if experiment.number in EXPERIMENTS:
        raise ValueError(f"실험 번호 {experiment.number}가 이미 등록되어 있습니다.")
    EXPERIMENTS[experiment.number] = experiment
    return experiment


def select(numbers):
    """번호 목록을 Experiment 목록으로 바꾼다. 순서를 유지하고 중복은 제거한다."""
    unknown = [n for n in numbers if n not in EXPERIMENTS]
    if unknown:
        available = ", ".join(map(str, sorted(EXPERIMENTS)))
        raise KeyError(f"등록되지 않은 실험: {unknown} (사용 가능: {available})")
    return [EXPERIMENTS[n] for n in dict.fromkeys(numbers)]

# ---------------------------------------------------------------- experiments
# 새 실험은 여기에 register(Experiment(...))로 추가한다.

# 교재 11.9 VGG-19-BN 설정: SGD(momentum=0.9), lr=0.001, batch_size=50, 5 epoch
VGG_OVERRIDES = {"epochs": 5, "batch_size": 50, "lr": 0.001,
                 "optimizer": "sgd", "optimizer_params": {"momentum": 0.9}}

# 실습문제 7번: 32×32 컬러 이미지를 64×64로 크게 변환 [3, 64, 64]
register(Experiment(
    number=1, key="vgg19_bn_64", title="VGG-19-BN Resize(64)",
    model=net.vgg19_bn,
    train_transform=partial(augment_transform, 64),
    test_transform=partial(base_transform, 64),
    overrides=VGG_OVERRIDES,
))

# 실습문제: 학습이 너무 느려서 112 -> 80으로 변경 (96도 가능)
register(Experiment(
    number=2, key="vgg19_bn_80", title="VGG-19-BN Resize(80)",
    model=net.vgg19_bn,
    train_transform=partial(augment_transform, 80),
    test_transform=partial(base_transform, 80),
    overrides=VGG_OVERRIDES,
))
