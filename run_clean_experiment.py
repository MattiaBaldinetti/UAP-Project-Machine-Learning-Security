# code/train.py
from __future__ import annotations

import argparse

from config import SEED, NUM_EPOCHS, NUM_CLASSES, CHECKPOINT_DIR
from utils import set_seed, get_device, ensure_dir
from data import get_cifar10_loaders
from training import train_and_eval_model

from models.my_resnet18 import ResNet18 as my_resnet18
from models.tv_resnet18 import TV_ResNet18_CIFAR10 as tv_resnet18
from models.densenet import densenet_cifar_light


def build_model(model_name: str, num_classes: int):
    """
    Ritorna una nuova istanza del modello richiesto.
    Nota: ogni chiamata crea un modello con pesi iniziali fresh.
    """
    if model_name == "my_resnet18":
        return my_resnet18()  # il tuo non accetta num_classes
    if model_name == "tv_resnet18":
        return tv_resnet18(num_classes=num_classes)
    if model_name == "densenet_light":
        return densenet_cifar_light(num_classes=num_classes)

    raise ValueError(f"Modello sconosciuto: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train one (or more) CIFAR-10 model(s) and save best checkpoint(s)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["my_resnet18", "tv_resnet18", "densenet_light", "all"],
        help="Which model to train. If omitted, trains 'all'.",
    )
    args = parser.parse_args()

    # Se non specifichi nulla, per praticità alleni tutto
    selected = args.model or "all"
    if selected == "all":
        models_to_train = ["my_resnet18", "tv_resnet18", "densenet_light"]
    else:
        models_to_train = [selected]

    # setup
    set_seed(SEED)
    device = get_device()
    print(f"[Setup] Device: {device}")

    ensure_dir(CHECKPOINT_DIR)

    train_loader, test_loader = get_cifar10_loaders(device=device)
    print(f"[Data] Train size: {len(train_loader.dataset)} | Test size: {len(test_loader.dataset)}")

    results = {}

    # training selettivo
    for name in models_to_train:
        set_seed(SEED)  # equità: stesso seed per ogni run modello
        model = build_model(name, NUM_CLASSES)

        model_trained, acc = train_and_eval_model(
            model, name, NUM_EPOCHS,
            train_loader, test_loader, device, CHECKPOINT_DIR
        )
        results[name] = acc

    print("\n========= TRAIN DONE =========")
    for name, acc in results.items():
        print(f"{name:15s} | TestAcc: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
