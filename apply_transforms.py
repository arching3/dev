import random

import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

DATA_ROOT = "./data"
OUTPUT_PATH = "outputs.png"


def main():
    # ./data 의 CIFAR-10 훈련 데이터셋 로드 (텐서로 변환)
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transforms.ToTensor()
    )

    # 무작위로 훈련 샘플 하나 선택
    idx = random.randrange(len(dataset))
    image, label = dataset[idx]
    class_name = dataset.classes[label]

    # 비교를 위해 p=1.0 으로 변환이 항상 적용되도록 설정
    flip = transforms.RandomHorizontalFlip(p=1.0)
    erasing = transforms.RandomErasing(
        p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    )
    both = transforms.Compose([flip, erasing])

    results = [
        ("Original", image),
        ("RandomHorizontalFlip", flip(image)),
        ("RandomErasing", erasing(image)),
        ("HorizontalFlip + Erasing", both(image)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img) in zip(axes, results):
        ax.imshow(img.permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Train sample #{idx} ({class_name})")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH} (index={idx}, class={class_name})")
    plt.show()


if __name__ == "__main__":
    main()
