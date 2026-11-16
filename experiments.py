"""실험 정의 레지스트리.

새 실험은 파일 하단에 register(Experiment(...))를 하나 추가하면 된다.
공통 값(optimizer 포함)은 config.py의 Config를 따르고, 실험마다 다른 값만 overrides에 적는다.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch.nn as nn
from torchvision import transforms

import net  # noqa: F401  (새 실험에서 net.<모델>로 사용)


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
# 새 실험은 여기에 register(Experiment(...))로 추가한다.

# 예시 (week10 실험 1~2). 사용하려면 net.py에 CNN_v2를 정의하고 주석을 해제한다.
# number는 train.py에 넘기는 실험 번호, key는 ckpts/·histories/ 아래 폴더 이름이다.
# 공통 값은 config.py를 따르고, 달라지는 값만 overrides에 적는다.
#
# # 실험 1: CNN_v2, 증강 없이 정규화만, SGD(lr=0.01), 30 epochs.
# register(Experiment(
#     number=1,
#     key="experiment1_cnn_v2",
#     title="Experiment 1: CNN_v2",
#     model=net.CNN_v2,
#     # BatchNorm 없는 CNN_v2 + 순수 SGD는 lr=0.001이면 사실상 학습되지 않아 lr=0.01을 쓴다.
#     overrides={"epochs": 30, "optimizer": "sgd", "lr": 0.01},
# ))
#
# # 실험 2: 같은 모델에 momentum=0.9 추가 (optimizer_params는 lr 이외 인자).
# register(Experiment(
#     number=2,
#     key="experiment2_cnn_v2",
#     title="Experiment 2: CNN_v2",
#     model=net.CNN_v2,
#     overrides={"epochs": 30, "optimizer": "sgd", "lr": 0.01,
#                "optimizer_params": {"momentum": 0.9}},
# ))
#
# # 증강을 쓰려면 train_transform=augment_transform 을 추가한다.
