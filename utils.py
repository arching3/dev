import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from pathlib import Path

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