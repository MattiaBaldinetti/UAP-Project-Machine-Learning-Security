# code/eval_uap.py
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from config import PROJECT_DIR, CIFAR10_MEAN, CIFAR10_STD
from load_models import load_and_freeze_model
from data import get_cifar10_loaders_pixelspace
from config import EPS_PIX


# -------------------------
# (A) Normalizer (stile prof)
# -------------------------
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def wrap_with_normalizer(model: nn.Module, device: torch.device) -> nn.Module:
    wrapped = nn.Sequential(Normalizer(CIFAR10_MEAN, CIFAR10_STD), model)
    wrapped.eval()
    return wrapped.to(device)


# -------------------------
# (B) Applica δ in pixel-space + clamp
# -------------------------
def apply_uap_to_batch(x_pix: torch.Tensor, delta_pix: torch.Tensor, eps_pix: float) -> torch.Tensor:
    """
    x_pix: [B,3,32,32] in [0,1]
    delta_pix: [1,3,32,32] oppure [3,32,32]
    """
    if delta_pix.dim() == 3:
        delta_pix = delta_pix.unsqueeze(0)  # -> [1,3,32,32]

    delta_pix = delta_pix.clamp(-eps_pix, eps_pix)     # safety
    x_adv = (x_pix + delta_pix).clamp(0.0, 1.0)        # clip immagine valida
    return x_adv


# -------------------------
# (C) Valutazione: clean acc, adv acc, fooling rate
# -------------------------
@torch.no_grad()
def evaluate_clean_and_uap(
    model_wrapped: nn.Module,
    loader_pix,
    delta_pix: torch.Tensor,
    eps_pix: float,
    device: torch.device
):
    model_wrapped.eval()

    total = 0
    correct_clean = 0
    correct_adv = 0
    fooled = 0

    delta_pix = delta_pix.to(device)

    for x_pix, y in loader_pix:
        x_pix = x_pix.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # CLEAN
        pred_clean = model_wrapped(x_pix).argmax(dim=1)

        # ADV
        x_adv = apply_uap_to_batch(x_pix, delta_pix, eps_pix)
        pred_adv = model_wrapped(x_adv).argmax(dim=1)

        total += y.size(0)
        correct_clean += (pred_clean == y).sum().item()
        correct_adv += (pred_adv == y).sum().item()
        fooled += (pred_adv != pred_clean).sum().item()

    acc_clean = correct_clean / total
    acc_adv = correct_adv / total
    fooling_rate = fooled / total
    return acc_clean, acc_adv, fooling_rate


def eps_to_str(eps_pix: float) -> str:
    return f"{eps_pix:.6f}".replace(".", "p")


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Setup] Device:", device)

    # carica e congela modelli
    my_model, _, _ = load_and_freeze_model("my")
    tv_model, _, _ = load_and_freeze_model("tv")

    # test loader pixel-space (NO Normalize)
    _, test_loader_pix = get_cifar10_loaders_pixelspace(device=device)

    # -------------------------
    # (D) Carica delta dai .pth in uaps/ (semplice come il notebook)
    # -------------------------
    eps_str = eps_to_str(EPS_PIX)

    uap_dir = Path(PROJECT_DIR) / "uaps"
    path_my = uap_dir / f"uap_my_resnet18_eps{eps_str}.pth"
    path_tv = uap_dir / f"uap_tv_resnet18_eps{eps_str}.pth"

    ckpt_my = torch.load(str(path_my), map_location="cpu")
    ckpt_tv = torch.load(str(path_tv), map_location="cpu")

    delta_my_pix = ckpt_my["delta_pix"]  # CPU
    delta_tv_pix = ckpt_tv["delta_pix"]  # CPU

    eps_my = float(ckpt_my.get("eps_pix", EPS_PIX))
    eps_tv = float(ckpt_tv.get("eps_pix", EPS_PIX))

    print("[UAP] Loaded:", path_my.name, "| eps:", eps_my)
    print("[UAP] Loaded:", path_tv.name, "| eps:", eps_tv)

    # -------------------------
    # (E) Wrappa e valuta 4 combinazioni (uguale al tuo)
    # -------------------------
    model_my_wrapped = wrap_with_normalizer(my_model, device)
    model_tv_wrapped = wrap_with_normalizer(tv_model, device)

    results = {}
    results["my_model + uap_my"] = evaluate_clean_and_uap(
        model_my_wrapped, test_loader_pix, delta_my_pix, eps_my, device
    )
    results["tv_model + uap_tv"] = evaluate_clean_and_uap(
        model_tv_wrapped, test_loader_pix, delta_tv_pix, eps_tv, device
    )
    results["tv_model + uap_my"] = evaluate_clean_and_uap(
        model_tv_wrapped, test_loader_pix, delta_my_pix, eps_my, device
    )
    results["my_model + uap_tv"] = evaluate_clean_and_uap(
        model_my_wrapped, test_loader_pix, delta_tv_pix, eps_tv, device
    )

    print("\n========= PUNTO 9: CLEAN vs ADV (UAP) =========")
    for k, (acc_c, acc_a, fr) in results.items():
        print(f"{k:20s} | clean: {acc_c*100:6.2f}% | adv: {acc_a*100:6.2f}% | fooling: {fr*100:6.2f}%")


if __name__ == "__main__":
    main()
