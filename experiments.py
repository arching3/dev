"""실험 정의 레지스트리.

새 실험은 파일 하단에 register(Experiment(...))를 하나 추가하면 된다.
공통 값(optimizer 포함)은 config.py의 Config를 따르고, 실험마다 다른 값만 overrides에 적는다.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch.nn as nn
from torchvision import transforms

from net import CNN_v2, CNN_v3, CNN_v4  # noqa: F401  (새 실험에서 바로 쓰도록 노출)


# ---------------------------------------------------------------- transforms
def base_transform():
    """증강 없이 텐서 변환 + 정규화만 적용한다."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])


def augment_transform():
    """RandomHorizontalFlip + RandomErasing 증강."""
    return transforms.Compose([
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
# 실험 1: CNN_v2, 증강 없이 정규화만, SGD, 30 epochs.
register(Experiment(
    number=1,
    key="experiment1_cnn_v2",
    title="Experiment 1: CNN_v2",
    model=CNN_v2,
    # BatchNorm 없는 CNN_v2 + 순수 SGD는 기울기가 작아 lr=0.001이면 사실상 학습되지 않는다 (노트북: lr=0.01).
    overrides={"epochs": 30, "optimizer": "sgd", "lr": 0.01},
))

# ---------------------------------------------------------------- experiments
# 실험 2: CNN_v2, 증강 없이 정규화만, SGD(lr=0.01)+momentum=0.9, 30 epochs.
register(Experiment(
    number=2,
    key="experiment2_cnn_v2",
    title="Experiment 2: CNN_v2",
    model=CNN_v2,
    overrides={"epochs": 30, "optimizer": "sgd", "lr": 0.01, "optimizer_params": {"momentum": 0.9}},
))

# ---------------------------------------------------------------- experiments
# 실험 3: CNN_v2, 증강 없이 정규화만, Adam, 30 epochs.
register(Experiment(
    number=3,
    key="experiment3_cnn_v2",
    title="Experiment 3: CNN_v2",
    model=CNN_v2,
    overrides={"epochs": 30, "optimizer": "adam"},
))

# ---------------------------------------------------------------- experiments
# 실험 4: CNN_v3, 증강 없이 정규화만, Adam, 30 epochs.
register(Experiment(
    number=4,
    key="experiment4_cnn_v3",
    title="Experiment 4: CNN_v3",
    model=CNN_v3,
    overrides={"epochs": 30},
))

# ---------------------------------------------------------------- experiments
# 실험 5: CNN_v4, 증강 없이 정규화만, Adam, 30 epochs.
register(Experiment(
    number=5,
    key="experiment5_cnn_v4",
    title="Experiment 5: CNN_v4",
    model=CNN_v4,
    overrides={"epochs": 30},
))

# ---------------------------------------------------------------- experiments
# 실험 6: CNN_v4, 증강 + RandomHorizontalFlip/RandomErasing, Adam, 30 epochs.
register(Experiment(
    number=6,
    key="experiment6_cnn_v4_aug",
    title="Experiment 6: CNN_v4 + augmentation",
    model=CNN_v4,
    train_transform=augment_transform,
    overrides={"epochs": 30, "optimizer": "adam"},
))