import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import (
    LR, MOMENTUM, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA,
    SEED, CIFAR10_MEAN, CIFAR10_STD,
)

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total

def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    running_loss, total = 0.0, 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total += y.size(0)

    return running_loss / total

def make_optim_and_sched(model: nn.Module):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )
    return optimizer, scheduler

def train_and_eval_model(
    model: nn.Module,
    name: str,
    num_epochs: int,
    train_loader,
    test_loader,
    device: torch.device,
    checkpoint_dir: Path | str,
):
    checkpoint_dir = Path(checkpoint_dir)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = make_optim_and_sched(model)

    best_acc = 0.0
    best_state = None

    print(f"\n==============================")
    print(f"Training model: {name}")
    print(f"==============================")

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_accuracy(model, test_loader, device)

        scheduler.step()

        print(f"{name} | Epoch {epoch:02d}/{num_epochs} | LR {lr_now:.5f} | Loss {train_loss:.4f} | TestAcc {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        # Caso limite: non è mai migliorato (non dovrebbe succedere, ma meglio essere robusti)
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    final_acc = evaluate_accuracy(model, test_loader, device)

    dt = time.time() - t0
    print(f"[{name}] BEST TestAcc: {final_acc*100:.2f}% | time: {dt/60:.1f} min")

    ckpt_path = checkpoint_dir / f"{name}_best.pth"
    torch.save(
        {
            "model_name": name,
            "model_state_dict": model.state_dict(),
            "best_test_acc": final_acc,
            "seed": SEED,
            "cifar10_mean": CIFAR10_MEAN,
            "cifar10_std": CIFAR10_STD,
            "hyperparams": {
                "epochs": num_epochs,
                "lr": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "lr_step_size": LR_STEP_SIZE,
                "lr_gamma": LR_GAMMA,
            },
        },
        ckpt_path,
    )
    print(f"[Saved checkpoint] {ckpt_path}")

    return model, final_acc
