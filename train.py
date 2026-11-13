"""CIFAR-10 실험 실행기.

예)
  python train.py 2          # 실험 2만
  python train.py 2 3        # 실험 2, 3을 한 번에 실행하고 비교 그래프 생성
  python train.py all        # 등록된 모든 실험
  python train.py --list     # 실험 목록
"""

import argparse
import ast
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

from config import CONFIG, get_device
from experiments import EXPERIMENTS, select
from utils import evaluate_history, fit


HISTORY_HEADER = "epoch,train_loss,train_accuracy,val_loss,val_accuracy"


def parse_param(text):
    """KEY=VALUE를 (key, value)로 변환한다. VALUE는 파이썬 리터럴(0.9, (0.9, 0.999), True)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"KEY=VALUE 형식이어야 합니다: {text}")
    try:
        return key.strip(), ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        return key.strip(), value.strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*",
                        help="실행할 실험 번호(여러 개 가능) 또는 all")
    parser.add_argument("--list", action="store_true", help="등록된 실험 목록 출력")
    # 아래 값은 지정했을 때만 config.py/실험별 설정을 덮어쓴다.
    parser.add_argument("--epochs", type=int, help="선택한 모든 실험의 에포크 일괄 변경")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--optimizer", help="optimizer 이름 (adam, adamw, sgd, rmsprop 또는 torch.optim 클래스명)")
    parser.add_argument("--optimizer-param", action="append", type=parse_param, default=[],
                        metavar="KEY=VALUE",
                        help="optimizer 인자, 반복 지정 가능 (예: momentum=0.9 weight_decay=5e-4)")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--show", action="store_true", help="학습 곡선을 화면에도 표시")
    args = parser.parse_args(argv)

    if args.list:
        return args, []
    if not args.experiments:
        parser.error("실험 번호를 하나 이상 지정하세요 (예: python train.py 2 3, --list로 목록 확인).")
    if "all" in args.experiments:
        numbers = sorted(EXPERIMENTS)
    else:
        try:
            numbers = [int(n) for n in args.experiments]
        except ValueError:
            parser.error("실험 번호는 정수 또는 all이어야 합니다.")
    try:
        selected = select(numbers)
    except KeyError as exc:
        parser.error(exc.args[0])
    return args, selected


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(experiment, cfg, device):
    """실험 하나를 학습하고 (N, 5) history를 반환한다. 결과는 output_dir/key/에 저장."""
    # 교재처럼 실험마다 같은 seed로 새 모델·optimizer·history를 구성한다.
    seed_everything(cfg.seed)
    train_set = datasets.CIFAR10(root=str(cfg.data_dir), train=True, download=True,
                                 transform=experiment.train_transform())
    test_set = datasets.CIFAR10(root=str(cfg.data_dir), train=False, download=True,
                                transform=experiment.test_transform())
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    tqdm.write(f"\n[{experiment.title}] Train: {len(train_set)}, Validation: {len(test_set)}, "
               f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}, "
               f"Optimizer: {cfg.optimizer_summary()}")
    net = experiment.model(len(train_set.classes)).to(device)
    criterion = experiment.criterion()
    optimizer = cfg.build_optimizer(net.parameters())
    history = fit(net, optimizer, criterion, cfg.epochs, train_loader, test_loader,
                  device, np.zeros((0, 5)))

    output_dir = cfg.output_dir / experiment.key
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "history.csv", history, delimiter=",", comments="",
               header=HISTORY_HEADER)
    torch.save(net.state_dict(), output_dir / "model_final.pth")
    return history


def main(argv=None):
    args, selected = parse_args(argv)
    if args.list:
        for number, exp in sorted(EXPERIMENTS.items()):
            cfg = exp.config(CONFIG)
            print(f"{number}: {exp.title} (model={exp.model.__name__}, "
                  f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
                  f"optimizer={cfg.optimizer_summary()})")
        return

    cli = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
           "seed": args.seed, "num_workers": args.num_workers,
           "data_dir": args.data_dir, "output_dir": args.output_dir,
           "optimizer": args.optimizer,
           "optimizer_params": dict(args.optimizer_param) or None}
    try:
        base = CONFIG.updated(**cli)
        configs = [experiment.config(CONFIG).updated(**cli) for experiment in selected]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"설정 오류: {exc}")
    device = get_device()
    print(f"Device: {device}", flush=True)

    histories = {}
    progress = tqdm(selected, desc="Experiments", position=0)
    # 우선순위: config.py < 실험별 overrides < 명령행 인자
    for experiment, cfg in zip(progress, configs):
        progress.set_description(experiment.title)
        histories[experiment.title] = run_experiment(experiment, cfg, device)

    # 선택한 실험들을 실험당 1행(손실·정확도 2열)으로 비교한다.
    evaluate_history(histories, output_dir=base.output_dir, show=args.show)
    print(f"결과 저장: {base.output_dir.resolve()}")


if __name__ == "__main__":
    main()
