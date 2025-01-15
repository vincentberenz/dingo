from typing import Any, Protocol

import torch


class TrainableModel(Protocol):
    network: torch.nn.Module
    device: torch.device

    def loss(self, theta: torch.Tensor, *context: torch.Tensor) -> torch.Tensor: ...


class TrainingInfo(Protocol):
    optimizer: torch.optim.Optimizer
    optimizer_kwargs: dict[str, Any]
    scheduler: torch.optimi.lr_scheduler.LRScheduler
    scheduler_kwargs: dict[str, Any]
