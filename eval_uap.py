from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

from config import PROJECT_DIR, CIFAR10_MEAN, CIFAR10_STD
from load_models import load_and_freeze_model
from data import get_cifar10_loaders_pixelspace


# -------------------------
# (A) Normalizer
# -------------------------
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
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
    if delta_pix.dim() == 3:
        delta_pix = delta_pix.unsqueeze(0)  # [1,3,32,32]
    delta_pix = delta_pix.clamp(-eps_pix, eps_pix)      # safety
    return (x_pix + delta_pix).clamp(0.0, 1.0)          # clip immagine valida


# -------------------------
# (C) Valutazione: clean acc, adv acc, fooling rate
# -------------------------
@torch.no_grad()
def evaluate_clean_and_uap(
    model_wrapped: nn.Module,
    loader_pix,
    delta_pix: torch.Tensor,
    eps_pix: float,
    device: torch.device,
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

        pred_clean = model_wrapped(x_pix).argmax(dim=1)

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


# -------------------------
# (D) Helpers path / parsing
# -------------------------
def _uaps_dir() -> Path:
    return Path(PROJECT_DIR) / "uaps"


def _resolve_uap_path(uap_arg: str) -> Path:
    p = Path(uap_arg)
    if p.exists() and p.is_file():
        return p
    candidate = _uaps_dir() / uap_arg
    if candidate.exists() and candidate.is_file():
        return candidate
    raise FileNotFoundError(f"UAP non trovata: '{uap_arg}' (né path né dentro {_uaps_dir()})")


def _normalize_model_name(name: str) -> str:
    name = name.strip().lower()
    aliases = {
        "tv_resnet": "tv_resnet18",
        "tv": "tv_resnet18",
        "my_resnet": "my_resnet18",
        "my": "my_resnet18",
        "dense": "densenet_light",
        "densenet": "densenet_light",
    }
    return aliases.get(name, name)


def _fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def _fmt_eps(eps: float) -> str:
    # 16/255 => "0.062745" (coerente col nome file)
    return f"{eps:.6f}"


def _load_uap(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    if "delta_pix" not in ckpt:
        raise KeyError(f"{path.name} non contiene la chiave 'delta_pix'")
    delta = ckpt["delta_pix"]
    eps = float(ckpt.get("eps_pix"))  # recupera il valore di eps dal checkpoint
    return delta, eps, ckpt

def _uap_base_name(path: Path) -> str:
    # rimuove estensione e suffisso _epsXpYYYYYY
    name = path.stem                     # uap_densenet_light_eps0p031373
    if "_eps" in name:
        name = name.split("_eps")[0]     # uap_densenet_light
    return name



# -------------------------
# (E) MAIN
# -------------------------
def main():
    args = sys.argv[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Setup] Device:", device)

    # loader test pixel-space (NO Normalize)
    _, test_loader_pix = get_cifar10_loaders_pixelspace(device=device)

    # Caso 2 argomenti: MODEL + UAP
    if len(args) == 2:
        model_name = _normalize_model_name(args[0])
        uap_path = _resolve_uap_path(args[1])

        model, _, _ = load_and_freeze_model(model_name, device=device)
        model_wrapped = wrap_with_normalizer(model, device)

        delta, eps_uap, _ = _load_uap(uap_path)
        acc_c, acc_a, fr = evaluate_clean_and_uap(model_wrapped, test_loader_pix, delta, eps_uap, device)

        print("\n========= Clean vs ADV (model × uap) =========")
        print(f"Valore di eps = {_fmt_eps(eps_uap)}")
        line = f"{model_name} + {_uap_base_name(uap_path):10s} | clean: {_fmt_pct(acc_c)} | adv: {_fmt_pct(acc_a)} | fooling: {_fmt_pct(fr)}"
        print(line)
        return

    # Caso 0 argomenti: TUTTI GLI INCROCI
    if len(args) != 0:
        raise SystemExit("Uso: python eval_uap.py [MODEL UAP_FILE]\n- senza argomenti: tutti gli incroci\n- con 2 argomenti: solo quel caso")

    # 1) trova tutte le UAP
    uap_dir = _uaps_dir()
    if not uap_dir.exists():
        raise FileNotFoundError(f"Cartella UAP non trovata: {uap_dir}")
    uap_paths = sorted(uap_dir.glob("*.pth"))
    if not uap_paths:
        raise FileNotFoundError(f"Nessun file .pth in: {uap_dir}")

    # 2) raggruppa UAP per eps
    uaps_by_eps: dict[float, list[Path]] = defaultdict(list)
    for p in uap_paths:
        _, eps_uap, _ = _load_uap(p)
        uaps_by_eps[eps_uap].append(p)

    # 3) modelli disponibili (quelli del tuo progetto)
    model_names = ["my_resnet18", "tv_resnet18", "densenet_light"]

    # 4) carica modelli una volta sola (più veloce)
    wrapped_models: dict[str, nn.Module] = {}
    for m in model_names:
        model, _, _ = load_and_freeze_model(m, device=device)
        wrapped_models[m] = wrap_with_normalizer(model, device)

    print("\n========= TUTTI I POSSIBILI Clean vs ADV (model × uap) =========")

    # 5) per ogni eps: stampa header + tutte le combinazioni
    for eps_uap in sorted(uaps_by_eps.keys()):
        print(f"\nValore di eps = {_fmt_eps(eps_uap)}")

        # pre-load delta per questo eps (eviti reload inutili)
        deltas = []
        for p in uaps_by_eps[eps_uap]:
            delta, _, _ = _load_uap(p)
            deltas.append((p, delta))

        # stampa tutte le righe: model × uap
        for model_name in model_names:
            model_wrapped = wrapped_models[model_name]
            for p, delta in deltas:
                acc_c, acc_a, fr = evaluate_clean_and_uap(
                    model_wrapped, test_loader_pix, delta, eps_uap, device
                )
                line = f"{model_name} + {_uap_base_name(p):10s} | clean: {_fmt_pct(acc_c)} | adv: {_fmt_pct(acc_a)} | fooling: {_fmt_pct(fr)}"
                print(line)


if __name__ == "__main__":
    main()
