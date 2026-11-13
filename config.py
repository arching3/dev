"""모든 실험이 공유하는 기본 하이퍼파라미터와 경로.

우선순위: Config 기본값 < 실험별 overrides(experiments.py) < train.py 명령행 인자
"""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.optim as optim


ROOT = Path(__file__).resolve().parent

# optimizer 이름 -> 클래스. 여기에 없는 이름도 torch.optim의 클래스명(예: "NAdam")이면 사용 가능.
OPTIMIZERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def resolve_optimizer(name) -> Callable[..., optim.Optimizer]:
    cls = OPTIMIZERS.get(name.lower()) or getattr(optim, name, None)
    if not (isinstance(cls, type) and issubclass(cls, optim.Optimizer)):
        raise ValueError(f"알 수 없는 optimizer: {name} (예: {', '.join(OPTIMIZERS)})")
    return cls


@dataclass(frozen=True)
class Config:
    epochs: int = 30
    batch_size: int = 100
    lr: float = 0.001
    seed: int = 123
    num_workers: int = 0
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "outputs"
    # optimizer 종류와 lr 이외의 인자(momentum, weight_decay, betas 등)
    optimizer: str = "adam"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.num_workers < 0:
            raise ValueError("epochs, batch_size, lr은 양수, num_workers는 0 이상이어야 합니다.")
        resolve_optimizer(self.optimizer)
        if "lr" in self.optimizer_params:
            raise ValueError("lr은 optimizer_params가 아니라 lr 항목으로 지정하세요.")

    def updated(self, **changes):
        """None이 아닌 값만 반영한 새 Config를 반환한다. 알 수 없는 키는 TypeError.

        optimizer_params는 optimizer가 그대로면 기존 값에 병합하고,
        optimizer가 바뀌면 새로 지정한 값으로 대체한다(이전 optimizer 인자는 버림).
        """
        names = {f.name for f in fields(self)}
        unknown = set(changes) - names
        if unknown:
            raise TypeError(f"알 수 없는 설정: {', '.join(sorted(unknown))}")
        changes = {k: v for k, v in changes.items() if v is not None}
        same_optimizer = changes.get("optimizer", self.optimizer).lower() == self.optimizer.lower()
        params = dict(self.optimizer_params) if same_optimizer else {}
        params.update(changes.get("optimizer_params", {}))
        changes["optimizer_params"] = params
        return replace(self, **changes)

    def build_optimizer(self, params):
        return resolve_optimizer(self.optimizer)(params, lr=self.lr, **self.optimizer_params)

    def optimizer_summary(self):
        extra = "".join(f", {k}={v}" for k, v in self.optimizer_params.items())
        return f"{self.optimizer}(lr={self.lr}{extra})"


CONFIG = Config()


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
