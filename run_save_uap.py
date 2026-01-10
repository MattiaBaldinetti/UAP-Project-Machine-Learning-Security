# code/run_uap.py
import os
from pathlib import Path
import torch

from config import PROJECT_DIR
from load_models import load_and_freeze_model
from data import get_cifar10_loaders_pixelspace
from uap import gen_uap_pixelspace

from config import (
    EPS_PIX,
    UAP_EPOCHS,
    STEP_DECAY,
    BETA,
    Y_TARGET,
)


def _eps_to_str(eps_pix: float) -> str:
    return f"{eps_pix:.6f}".replace(".", "p")


def save_uap_pth(
    delta_pix: torch.Tensor,
    model_name: str,
    eps_pix: float,
    *,
    beta: float,
    step_decay: float,
    uap_epochs: int,
    y_target,
    out_dir: Path,
    losses=None,
) -> Path:
    """
    Salva UAP in un .pth con metadati (pixel-space).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_str = _eps_to_str(eps_pix)
    uap_path = out_dir / f"uap_{model_name}_eps{eps_str}.pth"

    ckpt = {
        "delta_pix": delta_pix.detach().cpu(),   # portabile
        "space": "pixel",
        "eps_pix": float(eps_pix),
        "beta": float(beta),
        "step_decay": float(step_decay),
        "uap_epochs": int(uap_epochs),
        "targeted": (y_target is not None),
        "y_target": None if y_target is None else int(y_target),
        "trained_against": model_name,
        "shape": tuple(delta_pix.shape),
    }
    if losses is not None:
        ckpt["losses"] = list(losses)

    torch.save(ckpt, uap_path)
    return uap_path


def main():
    # 1) carica e congela modelli best
    my_model, _, device = load_and_freeze_model("my")
    tv_model, _, _ = load_and_freeze_model("tv")

    # 2) loader pixel-space (x in [0,1], NO Normalize)
    train_loader_pix, _ = get_cifar10_loaders_pixelspace(device=device)

    # 3) iperparametri UAP
    # Importati da config.py

    # 4) cartella output UAP (root/uaps)
    uap_dir = Path(PROJECT_DIR) / "uaps"

    # --- UAP contro my_model ---
    print("\n[UAP Pixel-space] Generating UAP for my_model...")
    delta_my_pix, losses_my = gen_uap_pixelspace(
        model=my_model,
        loader_pix=train_loader_pix,
        nb_epoch=UAP_EPOCHS,
        eps_pix=EPS_PIX,
        beta=BETA,
        step_decay=STEP_DECAY,
        y_target=Y_TARGET,
        delta_init=None,
        device=device,
    )

    path_my = save_uap_pth(
        delta_pix=delta_my_pix,
        model_name="my_resnet18",
        eps_pix=EPS_PIX,
        beta=BETA,
        step_decay=STEP_DECAY,
        uap_epochs=UAP_EPOCHS,
        y_target=Y_TARGET,
        out_dir=uap_dir,
        losses=losses_my,
    )
    print("[UAP SAVED] my_resnet18")
    print(" Path:", path_my)
    print("\n[UAP DONE] delta shape:", tuple(delta_my_pix.shape))
    print(" Delta min/max:", float(delta_my_pix.min()), float(delta_my_pix.max()))
    print("-" * 60)

    # --- UAP contro tv_model ---
    print("\n[UAP Pixel-space] Generating UAP for tv_model...")
    delta_tv_pix, losses_tv = gen_uap_pixelspace(
        model=tv_model,
        loader_pix=train_loader_pix,
        nb_epoch=UAP_EPOCHS,
        eps_pix=EPS_PIX,
        beta=BETA,
        step_decay=STEP_DECAY,
        y_target=Y_TARGET,
        delta_init=None,
        device=device,
    )

    path_tv = save_uap_pth(
        delta_pix=delta_tv_pix,
        model_name="tv_resnet18",
        eps_pix=EPS_PIX,
        beta=BETA,
        step_decay=STEP_DECAY,
        uap_epochs=UAP_EPOCHS,
        y_target=Y_TARGET,
        out_dir=uap_dir,
        losses=losses_tv,
    )
    print("[UAP SAVED] tv_resnet18")
    print(" Path:", path_tv)
    print("\n[UAP DONE] delta shape:", tuple(delta_tv_pix.shape))
    print(" Delta min/max:", float(delta_tv_pix.min()), float(delta_tv_pix.max()))
    print("-" * 60)


if __name__ == "__main__":
    main()
