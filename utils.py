import random

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter


def seed_everything(seed):
    """난수 고정 (교재 torch_seed 대체)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_loss(loader, device, net, criterion):
    """첫 미니배치의 손실을 계산한다. 손실 계산 그래프(make_dot) 시각화용."""
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    return criterion(net(images), labels)


def show_images_labels(loader, classes, net, device, n_images=50):
    """첫 미니배치의 이미지와 정답(:예측)을 표시한다. 오답은 파란색 제목."""
    images, labels = next(iter(loader))
    n_images = min(n_images, len(images))
    predicted = None
    if net is not None:
        net.eval()
        with torch.no_grad():
            predicted = net(images.to(device)).argmax(dim=1).cpu()

    plt.figure(figsize=(20, 15))
    for i in range(n_images):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        if predicted is None:
            ax.set_title(label_name, fontsize=20)
        else:
            predicted_name = classes[predicted[i]]
            color = "k" if label_name == predicted_name else "b"
            ax.set_title(f"{label_name}:{predicted_name}", c=color, fontsize=20)
        # Normalize(0.5, 0.5)를 되돌려 [0, 1] 범위로 표시
        image = (images[i].numpy().transpose(1, 2, 0) + 1) / 2
        ax.imshow(image.clip(0, 1))
        ax.set_axis_off()
    plt.show()


def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history,
        timing_history=None):
    """(N, 5) 기록 반환. timing_history에는 (epoch, train초, eval초)를 추가한다.

    시간은 데이터 로딩·장치 전송·연산·진행 표시를 포함한 단계 전체의 실측값이다.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs는 1 이상이어야 합니다.")
    if getattr(criterion, "reduction", "mean") != "mean":
        raise ValueError("criterion은 reduction='mean'을 사용해야 합니다.")
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 5:
        raise ValueError("history의 형태는 (N, 5)여야 합니다.")
    net.to(device)
    device = torch.device(device)
    timings = []
    base_epoch = int(history[-1, 0]) if len(history) else 0

    epochs = tqdm(range(base_epoch + 1, base_epoch + num_epochs + 1),
                  desc="Epochs", position=1, leave=False)
    for epoch in epochs:
        metrics = []
        durations = []
        for training, loader in ((True, train_loader), (False, test_loader)):
            net.train(training)
            total_loss, correct, count = 0.0, 0, 0
            phase = "Train" if training else "Validation"
            progress = tqdm(loader, desc=f"Epoch {epoch} {phase}",
                            position=2, leave=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(perf_counter() - started)
            if count == 0:
                raise ValueError(f"{phase} 데이터로더가 비어 있습니다.")
            metrics.extend((total_loss / count, correct / count))

        train_loss, train_acc, val_loss, val_acc = metrics
        epochs.set_postfix(loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.2%}")
        tqdm.write(f"Epoch [{epoch}/{base_epoch + num_epochs}] "
                   f"loss: {train_loss:.5f}, acc: {train_acc:.2%}, "
                   f"val_loss: {val_loss:.5f}, val_acc: {val_acc:.2%}, "
                   f"train: {durations[0]:.3f}s, eval: {durations[1]:.3f}s")
        history = np.vstack((history, [epoch, *metrics]))
        timings.append((epoch, *durations))
    if timing_history is not None:
        timing_history.extend(timings)
    means = np.asarray(timings)[:, 1:].mean(axis=0)
    tqdm.write(f"Average per epoch ({num_epochs} epochs): "
               f"train: {means[0]:.3f}s, eval: {means[1]:.3f}s")
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

    fig, axes = plt.subplots(len(histories), 2, figsize=(14, 6.5 * len(histories)),
                             squeeze=False, sharey="col")
    for row, (name, values) in enumerate(histories.items()):
        print(f"\n[{name}]")
        print(f"First epoch: validation loss {values[0, 3]:.5f}, validation accuracy {values[0, 4]:.2%}")
        print(f"Final epoch: validation loss {values[-1, 3]:.5f}, validation accuracy {values[-1, 4]:.2%}")
        best = values[np.argmax(values[:, 4])]
        print(f"Best validation accuracy: {best[4]:.2%} (epoch {int(best[0])})")
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
            # 영어 요약을 축 아래에 배치해 학습 곡선을 가리지 않는다.
            if metric == "Loss":
                summary = (
                    "Validation loss\n"
                    f"First epoch ({int(values[0, 0])}): {values[0, 3]:.5f}\n"
                    f"Final epoch ({int(values[-1, 0])}): {values[-1, 3]:.5f}"
                )
            else:
                summary = (
                    "Validation accuracy\n"
                    f"First: {values[0, 4]:.2%} | Final: {values[-1, 4]:.2%}\n"
                    f"Best: {best[4]:.2%} at epoch {int(best[0])}"
                )
                ax.scatter(best[0], best[4], color="tab:red", marker="*",
                           s=100, zorder=3)
            ax.text(0, -0.25, summary, transform=ax.transAxes,
                    ha="left", va="top", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f3f4f6",
                          "edgecolor": "#d1d5db"})
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "experiments.png")
    if show:
        plt.show()
    plt.close(fig)
