import datetime
from dataclasses import dataclass
import torch
import numpy as np
import mlflow

import logging

from model import Model
from mixins import EpochMixin


@dataclass
class TrainCustomModel:
    dataloader: torch.utils.data.DataLoader
    learning_rate: float
    epochs: int
    dropout: float
    save_all: bool
    clip_grad: float
    early_stopping_patience: int

    def __post_init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_params = Model(
            use_large=self.dataloader.use_large,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            save_all=self.save_all,
        )
        model_params.load_model(self.dataloader.labels)
        self.model_params = model_params
        self.epoch_params = EpochMixin()

    def get_sub_batches(self, batch):
        num_sample_batches = int(
            np.ceil(batch["input_ids"].shape[0] / self.dataloader.batch_size)
        )
        if num_sample_batches == 1:
            return [batch]
        else:
            sub_batches = []
            for i in range(num_sample_batches):
                empty_batch = {k: [] for k, v in batch.items()}
                for k, v in empty_batch.items():
                    empty_batch[k] = batch[k][
                        i * self.dataloader.batch_size : i * self.dataloader.batch_size
                        + self.dataloader.batch_size
                    ]
                sub_batches.append(empty_batch)
            return sub_batches

    def handle_batch_overflow(self, batch):
        loss = None
        sub_batches = self.get_sub_batches(batch)
        logging.info(f"Number of sub batches: {len(sub_batches)}")
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # in MB
        logging.info(f"GPU Memory Usage: {gpu_memory:.2f} MB")
        for sub_batch in sub_batches:
            for k, v in sub_batch.items():
                if k in ["idx"]:
                    continue
                sub_batch[k] = v.to(self.device)
            del sub_batch["idx"]

            try:
                outputs = self.model_params.model(**sub_batch)
            except Exception as e:
                logging.info(f"Exception: {e}")
                logging.info("#" * 40)

            if loss is None:
                loss = self.cross_entropy_loss(
                    logits=outputs.logits,
                    labels=sub_batch["labels"],
                    attention_mask=sub_batch["attention_mask"],
                )
            else:
                loss += self.cross_entropy_loss(
                    logits=outputs.logits,
                    labels=sub_batch["labels"],
                    attention_mask=sub_batch["attention_mask"],
                )

            logits = outputs.logits.detach().cpu()
            for k, v in sub_batch.items():
                sub_batch[k] = v.detach().cpu()

            self.epoch_params.batch_predictions = torch.cat(
                [self.epoch_params.batch_predictions, logits]
            )
            self.epoch_params.batch_labels = torch.cat(
                [self.epoch_params.batch_labels, sub_batch["labels"]]
            )
            self.epoch_params.batch_attention = torch.cat(
                [self.epoch_params.batch_attention, sub_batch["attention_mask"]]
            )
        return loss

    def cross_entropy_loss(self, logits, labels, attention_mask, size_average=True):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(
            logits, labels, reduction="none", label_smoothing=0.1
        )
        if attention_mask is not None:
            loss = loss * attention_mask.view(-1)
        if size_average:
            loss = loss.sum() / attention_mask.sum()
        return loss

    def train(self):
        self.model_params.model.to(self.device)
        self.model_params.start_time = datetime.datetime.now()

        print(f"TRAINING START TIME: {str(self.model_params.start_time)}", flush=True)
        print("=" * 40, flush=True)
        for epoch in range(self.model_params.epoch_num, self.epochs):

            #####################################################
            # TRAIN
            #####################################################

            self.model_params.model.train()
            self.epoch_params.reset_epoch()
            # self.model_params.epoch_list.append(epoch)
            print("#" * 40, flush=True)
            print(
                f"EPOCH START TIME: {str(self.epoch_params.epoch_time)}\tEPOCH NUMBER: {epoch}",
                flush=True,
            )
            for batch_idx, batch in enumerate(self.dataloader.train_dataloader):
                self.epoch_params.reset_batch()
                print("$" * 40, flush=True)
                print(
                    f"BATCH START TIME: {str(self.epoch_params.batch_time)}\tBATCH NUMBER: {batch_idx}",
                    flush=True,
                )

                # zero the parameter gradients
                self.model_params.optimizer.zero_grad()
                loss = self.handle_batch_overflow(batch)
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_params.model.parameters(), self.clip_grad
                    )
                self.model_params.optimizer.step()
                self.epoch_params.epoch_train_loss.append(loss.detach().item())
                print(
                    f"Loss after {self.model_params.train_steps} steps: {loss.detach().item()}, | time elapsed: {datetime.datetime.now()-self.model_params.start_time}",
                    flush=True,
                )
                self.model_params.train_steps += 1

                true_predictions = [
                    [
                        self.dataloader.id2label[torch.argmax(p).item()]
                        for (p, l) in zip(prediction, label)
                        if l != -100
                    ]
                    for prediction, label in zip(
                        self.epoch_params.batch_predictions,
                        self.epoch_params.batch_labels,
                    )
                ]
                true_labels = [
                    [
                        self.dataloader.id2label[l.item()]
                        for (p, l) in zip(prediction, label)
                        if l != -100
                    ]
                    for prediction, label in zip(
                        self.epoch_params.batch_predictions,
                        self.epoch_params.batch_labels,
                    )
                ]
                self.model_params.train_metric.add_batch(
                    predictions=true_predictions, references=true_labels
                )

                print(
                    f"TOTAL BATCH TIME: {datetime.datetime.now()-self.epoch_params.batch_time}",
                    flush=True,
                )
                print("$" * 40, flush=True)

            train_score = self.model_params.train_metric.compute()
            print("TRAIN SCORE = ", flush=True)
            for k, v in train_score.items():
                print(k, ":", v, flush=True)
            self.model_params.train_f1_scores.append(train_score["overall_f1"])

            train_loss = np.mean(self.epoch_params.epoch_train_loss)
            print("TRAIN LOSS = ", train_loss, flush=True)
            self.model_params.train_losses.append(train_loss)

            #####################################################
            # EVALUATE
            #####################################################

            self.model_params.model.eval()
            for batch in self.dataloader.test_dataloader:
                with torch.no_grad():
                    self.epoch_params.reset_batch()
                    loss = self.handle_batch_overflow(batch)
                    self.epoch_params.epoch_eval_loss.append(loss.detach().item())

                    true_predictions = [
                        [
                            self.dataloader.id2label[torch.argmax(p).item()]
                            for (p, l) in zip(prediction, label)
                            if l != -100
                        ]
                        for prediction, label in zip(
                            self.epoch_params.batch_predictions,
                            self.epoch_params.batch_labels,
                        )
                    ]
                    true_labels = [
                        [
                            self.dataloader.id2label[l.item()]
                            for (p, l) in zip(prediction, label)
                            if l != -100
                        ]
                        for prediction, label in zip(
                            self.epoch_params.batch_predictions,
                            self.epoch_params.batch_labels,
                        )
                    ]
                    self.model_params.eval_metric.add_batch(
                        predictions=true_predictions, references=true_labels
                    )

            eval_score = self.model_params.eval_metric.compute()
            print("EVAL SCORE = ", flush=True)
            for k, v in eval_score.items():
                print(k, ":", v, flush=True)
            self.model_params.eval_f1_scores.append(eval_score["overall_f1"])

            eval_loss = np.mean(self.epoch_params.epoch_eval_loss)
            print("EVAL LOSS = ", eval_loss, flush=True)
            self.model_params.eval_losses.append(eval_loss)

            if (
                eval_loss < self.model_params.last_best
                or eval_score["overall_f1"] > self.model_params.last_best_f1
            ) or (self.model_params.save_all):
                self.model_params.last_best = min(
                    self.model_params.last_best, eval_loss
                )
                self.model_params.last_best_f1 = max(
                    self.model_params.last_best_f1, eval_score["overall_f1"]
                )
                self.model_params.save_model(epoch, self.dataloader)

            self.model_params.scheduler.step(eval_loss)
            if (
                len(self.model_params.eval_losses) > self.early_stopping_patience
                and min(
                    self.model_params.eval_losses[-1 * self.early_stopping_patience :]
                )
                > self.model_params.last_best
            ):
                print("EARLY STOPPING", flush=True)
                break

            print(
                f"TOTAL EPOCH TIME: {datetime.datetime.now()-self.epoch_params.epoch_time}",
                flush=True,
            )
            print("#" * 40, flush=True)

        self.model_params.end_time = datetime.datetime.now()
        self.model_params.total_time = (
            self.model_params.end_time - self.model_params.start_time
        )

        for k, v in train_score.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in eval_score.items():
            mlflow.log_metric(f"eval_{k}", v)

        mlflow.log_metric("train_loss", self.model_params.train_losses[-1])
        mlflow.log_metric("eval_loss", self.model_params.eval_losses[-1])
        mlflow.log_metric("training_time", self.model_params.total_time)

        print(f"TOTAL TRAINING TIME: {self.model_params.total_time}", flush=True)
        print("=" * 40, flush=True)
