# code/train.py
from config import SEED, NUM_EPOCHS, NUM_CLASSES, CHECKPOINT_DIR
from utils import set_seed, get_device, ensure_dir
from data import get_cifar10_loaders
from training import train_and_eval_model

from models.my_resnet18 import ResNet18 as my_resnet18
from models.tv_resnet18 import TV_ResNet18_CIFAR10 as tv_resnet18

def main():
    set_seed(SEED)
    device = get_device()
    print(f"[Setup] Device: {device}")

    ensure_dir(CHECKPOINT_DIR)

    train_loader, test_loader = get_cifar10_loaders(device=device)
    print(f"[Data] Train size: {len(train_loader.dataset)} | Test size: {len(test_loader.dataset)}")

    set_seed(SEED)
    my_model, my_acc = train_and_eval_model(
        my_resnet18(), "my_resnet18", NUM_EPOCHS,
        train_loader, test_loader, device, CHECKPOINT_DIR
    )

    set_seed(SEED)
    tv_model, tv_acc = train_and_eval_model(
        tv_resnet18(NUM_CLASSES), "tv_resnet18", NUM_EPOCHS,
        train_loader, test_loader, device, CHECKPOINT_DIR
    )

    print("\n========= FINAL COMPARISON (CLEAN) =========")
    print(f"My ResNet18   TestAcc: {my_acc*100:.2f}%")
    print(f"TV ResNet18   TestAcc: {tv_acc*100:.2f}%")

if __name__ == "__main__":
    main()
