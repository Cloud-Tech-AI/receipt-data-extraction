import torch
import datetime


class TrainerMixin:
    def __init__(self):
        self.last_best = float("inf")
        self.last_best_f1 = -1.0
        self.train_steps = 0

        self.train_f1_scores = []
        self.train_losses = []
        self.eval_f1_scores = []
        self.eval_losses = []
        self.saved_files = []


class TimeMixin:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_time = None


class EpochMixin:
    def __init__(self):
        self.epoch_time = None
        self.batch_loss = None
        self.batch_time = None
        self.batch_predictions = None
        self.batch_labels = None
        self.batch_attention = None

    def reset_epoch(self):
        self.epoch_time = datetime.datetime.now()
        self.epoch_train_loss = []
        self.epoch_eval_loss = []

    def reset_batch(self):
        self.batch_time = datetime.datetime.now()

        self.batch_predictions = torch.tensor([]).type(torch.LongTensor)
        self.batch_labels = torch.tensor([]).type(torch.LongTensor)
        self.batch_attention = torch.tensor([]).type(torch.LongTensor)
