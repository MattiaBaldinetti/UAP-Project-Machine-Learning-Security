# code/load_models.py
from pathlib import Path
import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, NUM_CLASSES
from utils import get_device, freeze_model, count_trainable_params
from models.my_resnet18 import ResNet18 as my_resnet18
from models.tv_resnet18 import TV_ResNet18_CIFAR10 as tv_resnet18


def ckpt_path_for(model_name: str, checkpoint_dir: Path | str = CHECKPOINT_DIR) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir / f"{model_name}_best.pth"


def load_checkpoint_into_model(
    model: nn.Module,
    ckpt_path: Path | str,
    device: torch.device | str,
) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def load_and_freeze_model(kind: str):
    """
    kind: 'my' oppure 'tv'
    """
    device = get_device()

    if kind == "my":
        model_name = "my_resnet18"
        model = my_resnet18()
    elif kind == "tv":
        model_name = "tv_resnet18"
        model = tv_resnet18(num_classes=NUM_CLASSES)
    else:
        raise ValueError("kind must be 'my' or 'tv'")

    model = model.to(device)
    ckpt_path = ckpt_path_for(model_name)

    ckpt = load_checkpoint_into_model(model, ckpt_path, device=device)
    print(f"[Load] {ckpt_path.name} --> best_test_acc:", ckpt.get("best_test_acc", None))

    model = freeze_model(model)
    trainable = count_trainable_params(model)
    print(f"[Freeze] {model_name} trainable params:", trainable)

    assert trainable == 0
    return model, ckpt, device


# ---- opzionale: test standalone ----
def main():
    print("Device:", get_device())
    load_and_freeze_model("my")
    load_and_freeze_model("tv")
    print("[OK] Modelli caricati e congelati.")


if __name__ == "__main__":
    main()
