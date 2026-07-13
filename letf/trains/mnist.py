"""Builtin MNIST training demo."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from letf.context import TrainContext, TrainResult


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(ctx: TrainContext) -> TrainResult:
    """Train a small MLP on MNIST; logs loss each epoch."""
    hp = ctx.hparams
    epochs = int(hp.get("epochs", 1))
    batch_size = int(hp.get("batch_size", 64))
    lr = float(hp.get("lr", 0.01))
    max_batches = hp.get("max_batches")  # optional; limits batches per epoch
    data_dir = str(hp.get("data_dir", ctx.run_dir / "data"))

    torch.manual_seed(ctx.config.seed)
    device = torch.device(ctx.device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _MLP().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    last_loss = 0.0
    global_step = 0

    for epoch in range(epochs):
        running = 0.0
        n = 0
        for batch_idx, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

            last_loss = float(loss.item())
            running += last_loss
            n += 1
            global_step += 1
            ctx.log_metric(step=global_step, loss=last_loss, epoch=epoch)

        avg = running / max(n, 1)
        ctx.log_metric(step=global_step, epoch=epoch, epoch_loss=avg)

    model_path = ctx.run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    return TrainResult(
        metrics={"final_loss": last_loss, "epochs": epochs},
        artifacts={"model": str(model_path)},
    )