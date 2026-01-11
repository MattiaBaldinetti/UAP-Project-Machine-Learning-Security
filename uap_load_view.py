from __future__ import annotations

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from config import PROJECT_DIR, EPS_PIX
from data import get_cifar10_loaders_pixelspace


def _uaps_dir() -> Path:
    return Path(PROJECT_DIR) / "uaps"


def _uap_img_dir() -> Path:
    d = Path(PROJECT_DIR) / "uap_img"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _eps_to_str(eps_pix: float) -> str:
    return f"{eps_pix:.6f}".replace(".", "p")


def _resolve_uap_path(file_arg: str | None) -> Path | None:
    if file_arg is None:
        return None

    p = Path(file_arg)
    if p.exists() and p.is_file():
        return p

    candidate = _uaps_dir() / file_arg
    if candidate.exists() and candidate.is_file():
        return candidate

    raise FileNotFoundError(f"File UAP non trovato: '{file_arg}'.")


def load_uap_from_path(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu")
    if "delta_pix" not in ckpt:
        raise KeyError(f"{path.name} non contiene 'delta_pix'")
    return ckpt["delta_pix"].to(device), ckpt


# -------------------------
# Figura 1: visualizzazione δ
# -------------------------
def show_uap(delta_pix: torch.Tensor, eps_pix: float, model_name: str):
    out_dir = _uap_img_dir()
    eps_str = _eps_to_str(eps_pix)

    d = delta_pix.detach().squeeze(0).cpu()  # [3,32,32]

    # δ mapped [-eps,+eps] -> [0,1]
    d_vis = (d / eps_pix).clamp(-1, 1)
    d_vis = (d_vis + 1) / 2
    d_vis_img = d_vis.permute(1, 2, 0).numpy()

    # δ amplified (solo display)
    amp = 10.0
    d_amp = (d * amp / eps_pix).clamp(-1, 1)
    d_amp = (d_amp + 1) / 2
    d_amp_img = d_amp.permute(1, 2, 0).numpy()

    # heatmap L2 per pixel
    heat = torch.sqrt((d ** 2).sum(dim=0)).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(d_vis_img)
    plt.title(f"{model_name} UAP δ mapped [-ε,ε]→[0,1]")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(d_amp_img)
    plt.title(f"{model_name} UAP δ amplified (x{amp})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(heat)
    plt.title(f"{model_name} |δ| heat (L2 per-pixel)")
    plt.axis("off")

    plt.tight_layout()

    out_path = out_dir / f"uap_lambda_{model_name}_eps{eps_str}.png"
    plt.savefig(out_path, dpi=200)
    print("[IMG SAVED]", out_path)

    plt.show()
    plt.close()


# -------------------------
# Figura 2: esempio immagine
# -------------------------
def show_example_perturbation(
    delta_pix: torch.Tensor,
    eps_pix: float,
    model_name: str,
    device: torch.device,
):
    out_dir = _uap_img_dir()
    eps_str = _eps_to_str(eps_pix)

    _, test_loader_pix = get_cifar10_loaders_pixelspace(device=device)
    xb_pix, _ = next(iter(test_loader_pix))

    x = xb_pix[0:1].to(device, non_blocking=True)  # [1,3,32,32]
    x_pert = torch.clamp(x + delta_pix.to(device), 0.0, 1.0)

    x_img = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    xpert_img = x_pert.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    diff = (x_pert - x).detach().cpu().squeeze(0)
    diff_vis = (diff / eps_pix).clamp(-1, 1)
    diff_vis = (diff_vis + 1) / 2
    diff_img = diff_vis.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(x_img)
    plt.title("Original (pixel-space)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(xpert_img)
    plt.title("Perturbed (pixel-space)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff_img)
    plt.title("Δ (mapped to [0,1])")
    plt.axis("off")

    plt.tight_layout()

    out_path = out_dir / f"image_perturbation_{model_name}_eps{eps_str}.png"
    plt.savefig(out_path, dpi=200)
    print("[IMG SAVED]", out_path)

    plt.show()
    plt.close()


def visualize_one(path: Path, device: torch.device):
    delta, ckpt = load_uap_from_path(path, device)
    eps_pix = float(ckpt.get("eps_pix", EPS_PIX))

    trained_against = ckpt.get("trained_against", "uap")
    if trained_against == "tv_resnet18":
        model_name = "tv_resnet18"
    elif trained_against == "my_resnet18":
        model_name = "my_resnet18"
    else:
        model_name = str(trained_against)

    # figura 1: UAP (λ / δ)
    show_uap(delta, eps_pix, model_name)

    # figura 2: esempio immagine perturbata
    show_example_perturbation(delta, eps_pix, model_name, device)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza (e salva) le immagini della UAP. "
                    "Se non passi nulla, visualizza tutte le .pth in uaps/. "
                    "Se passi un nome file/path, visualizza solo quello."
    )
    
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Nome file .pth (in uaps/) oppure path completo. Se omesso: tutti.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Setup] Device:", device)

    if args.file is not None:
        path = _resolve_uap_path(args.file)
        visualize_one(path, device)
        return

    uap_dir = _uaps_dir()
    paths = sorted(uap_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"Nessun file .pth trovato in: {uap_dir}")

    for p in paths:
        visualize_one(p, device)


if __name__ == "__main__":
    main()