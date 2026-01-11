from __future__ import annotations
import argparse
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

# --- Modelli supportati ---
# Chiave = quello che scrivi da CLI
# Valore = quello che passa a load_and_freeze_model(...)
MODEL_ALIASES = {
    "my_resnet18": "my",          # oppure "my_resnet18"
    "tv_resnet18": "tv",          # oppure "tv_resnet18"
    "densenet_light": "densenet", # oppure "densenet_light"
}


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
    parser = argparse.ArgumentParser(
        description=(
            "Generate and save pixel-space UAP(s).\n"
            "Uso:\n"
            "  python run_save_uap.py                # tutti i modelli\n"
            "  python run_save_uap.py densenet_light # solo un modello\n"
        )
    )

    # UN SOLO ARGOMENTO POSIZIONALE (opzionale)
    parser.add_argument(
        "model",
        nargs="?",                 # opzionale
        default="all",
        choices=list(MODEL_ALIASES.keys()) + ["all"],
        help="Model name (default: all).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Setup] Device:", device)

    # quali modelli processare
    if args.model == "all":
        selected_models = list(MODEL_ALIASES.keys())
    else:
        selected_models = [args.model]

    # loader pixel-space (x in [0,1], NO Normalize)
    train_loader_pix, _ = get_cifar10_loaders_pixelspace(device=device)

    # output dir UAP
    uap_dir = Path(PROJECT_DIR) / "uaps"
    uap_dir.mkdir(parents=True, exist_ok=True)

    for model_name in selected_models:
        print("\n==============================")
        print(f"[UAP Pixel-space] Generating UAP for {model_name}...")
        print("==============================")

        # carica e congela solo quel modello
        model, _, _ = load_and_freeze_model(model_name, device=device)

        delta_pix, losses = gen_uap_pixelspace(
            model=model,
            loader_pix=train_loader_pix,
            nb_epoch=UAP_EPOCHS,
            eps_pix=EPS_PIX,
            beta=BETA,
            step_decay=STEP_DECAY,
            y_target=Y_TARGET,
            delta_init=None,
            device=device,
        )

        path = save_uap_pth(
            delta_pix=delta_pix,
            model_name=model_name,   # nome “standard” usato nel filename
            eps_pix=EPS_PIX,
            beta=BETA,
            step_decay=STEP_DECAY,
            uap_epochs=UAP_EPOCHS,
            y_target=Y_TARGET,
            out_dir=uap_dir,
            losses=losses,
        )

        print(f"[UAP SAVED] {model_name}")
        print(" Path:", path)
        print(f"[UAP DONE] Delta shape:", tuple(delta_pix.shape))
        print(" Delta min/max:", float(delta_pix.min()), float(delta_pix.max()))
        print("-" * 60)


if __name__ == "__main__":
    main()