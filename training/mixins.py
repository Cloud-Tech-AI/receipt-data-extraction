from dataclasses import dataclass, field
from typing import Optional
import torch
import datetime


@dataclass
class TrainerMixin:
    last_best: float = float("inf")
    last_best_f1: float = -1.0
    train_steps: int = 0

    train_f1_scores: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    eval_f1_scores: list = field(default_factory=list)
    eval_losses: list = field(default_factory=list)
    saved_files: list = field(default_factory=list)


@dataclass
class TimeMixin:
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    total_time: datetime.datetime = None


class EpochMixin:
    epoch_time: datetime.datetime = None
    epoch_train_loss: Optional[list] = None
    epoch_eval_loss: Optional[list] = None

    batch_loss: Optional[float] = None
    batch_time: Optional[datetime.datetime] = None
    batch_predictions: Optional[torch.Tensor] = None
    batch_labels: Optional[torch.Tensor] = None
    batch_attention: Optional[torch.Tensor] = None

    def reset_epoch(self):
        self.epoch_time = datetime.datetime.now()
        self.epoch_train_loss = []
        self.epoch_eval_loss = []

    def reset_batch(self):
        self.batch_time = datetime.datetime.now()

        self.batch_predictions = torch.tensor([], dtype=torch.long)
        self.batch_labels = torch.tensor([], dtype=torch.long)
        self.batch_attention = torch.tensor([], dtype=torch.long)
