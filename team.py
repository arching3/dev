"""CIFAR-10 실험 2·3 실행: python team.py (교재의 50·100 에포크 적용)."""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

class CNN_v4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.conv3,
            self.bn3,
            self.relu,
            self.conv4,
            self.bn4,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.conv5,
            self.bn5,
            self.relu,
            self.conv6,
            self.bn6,
            self.relu,
            self.maxpool,
            self.dropout3,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3


def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    """노트북과 같은 (N, 5) 학습 기록을 반환한다. criterion은 mean reduction 사용."""
    if num_epochs < 1:
        raise ValueError("num_epochs는 1 이상이어야 합니다.")
    if getattr(criterion, "reduction", "mean") != "mean":
        raise ValueError("criterion은 reduction='mean'을 사용해야 합니다.")
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 5:
        raise ValueError("history의 형태는 (N, 5)여야 합니다.")
    net.to(device)
    base_epoch = int(history[-1, 0]) if len(history) else 0

    epochs = tqdm(range(base_epoch + 1, base_epoch + num_epochs + 1),
                  desc="Epochs", position=1, leave=False)
    for epoch in epochs:
        metrics = []
        for training, loader in ((True, train_loader), (False, test_loader)):
            net.train(training)
            total_loss, correct, count = 0.0, 0, 0
            phase = "Train" if training else "Validation"
            progress = tqdm(loader, desc=f"Epoch {epoch} {phase}",
                            position=2, leave=False)
            with torch.set_grad_enabled(training):
                for inputs, labels in progress:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if training:
                        optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    count += batch_size
                    progress.set_postfix(loss=f"{total_loss / count:.4f}",
                                         accuracy=f"{correct / count:.2%}")
            if count == 0:
                raise ValueError(f"{phase} 데이터로더가 비어 있습니다.")
            metrics.extend((total_loss / count, correct / count))

        train_loss, train_acc, val_loss, val_acc = metrics
        epochs.set_postfix(loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.2%}")
        tqdm.write(f"Epoch [{epoch}/{base_epoch + num_epochs}] "
                   f"loss: {train_loss:.5f}, acc: {train_acc:.2%}, "
                   f"val_loss: {val_loss:.5f}, val_acc: {val_acc:.2%}")
        history = np.vstack((history, [epoch, *metrics]))
    return history


def evaluate_history(history, output_dir=None, show=True):
    """단일 기록 또는 {실험명: 기록}을 받아 실험별 행에 손실·정확도를 그린다."""
    histories = history if isinstance(history, dict) else {"Training": history}
    if not histories:
        raise ValueError("학습 기록이 없습니다.")
    histories = {name: np.asarray(values, dtype=float)
                 for name, values in histories.items()}
    for values in histories.values():
        if values.ndim != 2 or values.shape[1] != 5 or len(values) == 0:
            raise ValueError("각 history는 비어 있지 않은 (N, 5) 배열이어야 합니다.")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(histories), 2, figsize=(14, 5 * len(histories)),
                             squeeze=False, sharey="col")
    for row, (name, values) in enumerate(histories.items()):
        print(f"\n[{name}]")
        print(f"첫 에포크: 검증 손실 {values[0, 3]:.5f}, 검증 정확도 {values[0, 4]:.2%}")
        print(f"최종 상태: 검증 손실 {values[-1, 3]:.5f}, 검증 정확도 {values[-1, 4]:.2%}")
        best = values[np.argmax(values[:, 4])]
        print(f"최고 검증 정확도: {best[4]:.2%} (epoch {int(best[0])})")
        for column, (metric, train_column, val_column) in enumerate(
                (("Loss", 1, 3), ("Accuracy", 2, 4))):
            ax = axes[row, column]
            ax.plot(values[:, 0], values[:, train_column], label="Train")
            ax.plot(values[:, 0], values[:, val_column], label="Validation")
            ax.set(xlabel="Epoch", ylabel=metric, title=f"{name} - {metric}")
            if metric == "Accuracy":
                ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "experiments.png")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None,
                        help="두 실험의 에포크를 일괄 변경 (기본: 실험 2=50, 실험 3=100)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--show", action="store_true", help="학습 곡선을 화면에도 표시")
    args = parser.parse_args(argv)
    if (args.epochs is not None and args.epochs < 1) or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, lr은 양수여야 합니다.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3),
                                 value=0, inplace=False),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
    ])
    test_set = datasets.CIFAR10(root=str(args.data_dir), train=False,
                               download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    experiments = [
        # 실험 2: CNN_v4 (Dropout + BatchNorm), 증강 없이 정규화만, Adam, 50 epochs.
        ("experiment2_cnn_v4", "Experiment 2: CNN_v4", transform_test, 50),
        # 실험 3: CNN_v4 + RandomHorizontalFlip/RandomErasing, Adam, 100 epochs.
        ("experiment3_cnn_v4_aug", "Experiment 3: CNN_v4 + augmentation", transform_train, 100),
    ]
    histories = {}
    progress = tqdm(experiments, desc="Experiments", position=0)
    for experiment_id, name, transform, default_epochs in progress:
        progress.set_description(name)
        # 교재처럼 실험마다 같은 seed로 새 모델·optimizer·history를 구성한다.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        train_set = datasets.CIFAR10(root=str(args.data_dir), train=True,
                                    download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        num_epochs = args.epochs if args.epochs is not None else default_epochs
        tqdm.write(f"\n[{name}] Train: {len(train_set)}, Validation: {len(test_set)}, "
                   f"Epochs: {num_epochs}, Batch size: {args.batch_size}, Adam lr: {args.lr}")
        net = CNN_v4(len(train_set.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        history = fit(net, optimizer, criterion, num_epochs, train_loader,
                      test_loader, device, np.zeros((0, 5)))
        histories[name] = history

        # 실험별 출력 디렉터리로 학습 기록과 최종 가중치를 구분한다.
        output_dir = args.output_dir / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "history.csv", history, delimiter=",", comments="",
                   header="epoch,train_loss,train_accuracy,val_loss,val_accuracy")
        torch.save(net.state_dict(), output_dir / "model_final.pth")
        del optimizer, net

    # 공통 evaluate_history에서 실험 2·3을 2행 × 2열 subplot으로 비교한다.
    evaluate_history(histories, output_dir=args.output_dir, show=args.show)
    print(f"결과 저장: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()


