from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from data import normalize_batch


def project_delta_linf_pix_(delta: torch.Tensor, eps_pix: float):
    with torch.no_grad():
        delta.clamp_(-eps_pix, eps_pix)


def apply_uap_pixelspace(x_pix: torch.Tensor,
                         delta_pix: torch.Tensor,
                         pixel_clip: bool = True) -> torch.Tensor:
    x_pert = x_pix + delta_pix
    if pixel_clip:
        x_pert = torch.clamp(x_pert, 0.0, 1.0)
    return x_pert


def make_clamped_loss(beta: float, device: torch.device):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    beta_t = torch.tensor(beta, device=device, dtype=torch.float32)

    def clamped_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raw = loss_fn(outputs, targets)
        clamped = torch.min(raw, beta_t)
        return clamped.mean()

    return clamped_loss


def gen_uap_pixelspace(
    model: nn.Module,
    loader_pix,
    nb_epoch: int,
    eps_pix: float,
    beta: float = 12.0,
    step_decay: float = 0.8,
    y_target: Optional[int] = None,
    delta_init: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    UAP in pixel-space:
      x_pert_pix = clip(x_pix + δ, 0, 1)
      x_pert_norm = normalize_batch(x_pert_pix)
      forward sul modello (fisso), aggiorni SOLO δ (stile L∞ sign gradient)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    # inferisci shape
    x0, _ = next(iter(loader_pix))
    x0 = x0.to(device)

    if delta_init is None:
        delta_pix = torch.zeros(1, *x0.shape[1:], device=device, dtype=torch.float32)
    else:
        d = delta_init.detach().to(device).float()
        if d.dim() == 3:
            d = d.unsqueeze(0)
        delta_pix = d

    # proiezione iniziale L∞
    delta_pix = torch.clamp(delta_pix, -eps_pix, eps_pix).detach()

    clamped_loss = make_clamped_loss(beta=beta, device=device)
    losses: List[float] = []

    for epoch in range(nb_epoch):
        step = eps_pix * (step_decay ** epoch)
        print(f"[UAP-Pix] Epoch {epoch+1}/{nb_epoch} | step {step:.6f} | eps {eps_pix:.6f}")

        for xb_pix, yb in loader_pix:
            xb_pix = xb_pix.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if y_target is not None:
                yb = torch.full_like(yb, fill_value=y_target, device=device)

            batch_delta = delta_pix.expand(xb_pix.size(0), -1, -1, -1).detach()
            batch_delta.requires_grad_(True)

            x_pert_pix = torch.clamp(xb_pix + batch_delta, 0.0, 1.0)
            x_pert_norm = normalize_batch(x_pert_pix)
            logits = model(x_pert_norm)

            loss = clamped_loss(logits, yb)
            if y_target is not None:
                loss = -loss

            grad = torch.autograd.grad(loss, batch_delta, retain_graph=False, create_graph=False)[0]
            grad_sign = grad.mean(dim=0, keepdim=True).sign()

            delta_pix = (delta_pix + step * grad_sign).clamp(-eps_pix, eps_pix).detach()
            losses.append(float(loss.detach().item()))

        if len(loader_pix) > 0:
            last = losses[-len(loader_pix):]
            print(f"  epoch mean loss: {sum(last)/len(last):.4f} | delta min/max: {delta_pix.min().item():.4f}/{delta_pix.max().item():.4f}")

    return delta_pix, losses
    