from __future__ import annotations
from pathlib import Path
import argparse
import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, NUM_CLASSES
from utils import get_device, freeze_model, count_trainable_params

from models.my_resnet18 import ResNet18 as my_resnet18
from models.tv_resnet18 import TV_ResNet18_CIFAR10 as tv_resnet18
from models.densenet import densenet_cifar_light


def ckpt_path_for(model_name: str, checkpoint_dir: Path | str = CHECKPOINT_DIR) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir / f"{model_name}_best.pth"


def load_checkpoint_into_model(
    model: nn.Module,
    ckpt_path: Path | str,
    device: torch.device | str,
) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{Path(ckpt_path).name} non contiene 'model_state_dict'.")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Crea una nuova istanza del modello richiesto (pesi fresh).
    """
    if model_name == "my_resnet18":
        return my_resnet18()
    if model_name == "tv_resnet18":
        return tv_resnet18(num_classes=num_classes)
    if model_name == "densenet_light":
        return densenet_cifar_light(num_classes=num_classes)

    raise ValueError(f"Modello sconosciuto: {model_name}")


def load_and_freeze_model(model_name: str, device: torch.device | None = None):
    """
    Carica <CHECKPOINT_DIR>/<model_name>_best.pth in un modello appena costruito
    e lo congela (eval + requires_grad=False).

    Ritorna: (model, ckpt, device)
    """
    if device is None:
        device = get_device()

    ckpt_path = ckpt_path_for(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    model = build_model(model_name, NUM_CLASSES).to(device)

    ckpt = load_checkpoint_into_model(model, ckpt_path, device=device)
    print(f"[Load] {ckpt_path.name} --> best_test_acc:", ckpt.get("best_test_acc", None))

    model = freeze_model(model)
    trainable = count_trainable_params(model)
    print(f"[Freeze] {model_name} trainable params:", trainable)
    assert trainable == 0

    return model, ckpt, device
    

def main():
    parser = argparse.ArgumentParser(
        description="Load & freeze trained model checkpoints. Default: load ALL."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["my_resnet18", "tv_resnet18", "densenet_light", "all"],
        help="Which model to load. If omitted, loads all.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    if args.model == "all":
        model_names = ["my_resnet18", "tv_resnet18", "densenet_light"]
    else:
        model_names = [args.model]

    loaded = {}
    for name in model_names:
        model, ckpt, _ = load_and_freeze_model(name, device=device)
        loaded[name] = (model, ckpt)

    print("\n[OK] Modelli caricati e congelati:")
    for name in loaded:
        print(" -", name)


if __name__ == "__main__":
    main()
