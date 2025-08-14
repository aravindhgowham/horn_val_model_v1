import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from dataset import RawAudioFolderDataset, AudioPreprocConfig
from model import RawAudioCNN1D
from utils import seed_everything, compute_class_weights, save_yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress size mismatch warnings


# Load environment variables from .env if present
load_dotenv()
ENV_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
ENV_DURATION_SEC = float(os.getenv("AUDIO_DURATION_SEC", "4.0"))

class parse_args():
    train_dir = "data/train"
    val_dir = "data/val"
    output_dir = "checkpoints"
    sample_rate = ENV_SAMPLE_RATE
    duration_sec = ENV_DURATION_SEC
    batch_size = 32 #32
    epochs = 30
    lr = 1e-3 #1e - 3
    weight_decay = 1e-4 #1e - 4
    num_workers = 4
    seed = 42
    base_channels = 32
    dropout = 0.1
    use_amp = False



def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)

    preproc = AudioPreprocConfig(
        target_sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        random_crop=True,
        normalize=True,
    )
    
    print("\nPreproc: ",preproc)
    # Datasets
    train_ds = RawAudioFolderDataset(
        root_dir=args.train_dir,
        preproc=preproc,
        class_name_to_index={"fail": 0, "pass": 1},
        augment=True,
    )
    val_ds = RawAudioFolderDataset(
        root_dir=args.val_dir,
        preproc=AudioPreprocConfig(
            target_sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            random_crop=False,
            normalize=True,
        ),
        class_name_to_index=train_ds.get_label_mapping(),
        augment=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawAudioCNN1D(in_channels=1, num_classes=2, base_channels=args.base_channels, dropout=args.dropout)
    model.to(device)

    # Class weights for imbalance
    class_weights = compute_class_weights(train_ds.class_distribution()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_parameter = sum(p.numel() for p in model.parameters())
    print(f"\nModel Capacity:\nTotal parameters: {total_parameter}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    # Save label map for inference
    label_map_path = os.path.join(args.output_dir, "label_map.yaml")
    save_yaml(train_ds.get_label_mapping(), label_map_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits.detach(), dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            pbar.set_postfix({
                "train_loss": f"{running_loss / max(running_total, 1):.4f}",
                "train_acc": f"{running_correct / max(running_total, 1):.4f}",
            })

        scheduler.step()

        # Validation
        metrics = evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        print(f"\nValidation - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map_path": label_map_path,
                "args": vars(args),
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_ckpt_path)
            print(f"Saved best checkpoint to {best_ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()