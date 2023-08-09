import traceback
import datetime
import copy
import torch
import numpy as np

from model import Model
from mixins import EpochMixin


class TrainCustomModel:
    def __init__(self,
                 dataloader,
                 concurrency,
                 learning_rate,
                 epochs,
                 dropout,
                 save_all,
                 artefact_dir,
                 clip_grad,
                 early_stopping_patience,
                 ):
        self.dataloader = dataloader
        self.concurrency = concurrency
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model_params = Model(use_large=dataloader.use_large, learning_rate=learning_rate, dropout=dropout, save_all=save_all, artefact_dir=artefact_dir)
        model_params.load_model(self.dataloader.labels)
        self.model_params = model_params
        self.epoch_params = EpochMixin()

    def group_samples(self, batch):
        num_sample_groups = int(
            np.ceil(batch["input_ids"].shape[0]/self.concurrency))
        if num_sample_groups == 1:
            return [batch]
        else:
            sample_groups = []
            for i in range(num_sample_groups):
                empty_batch = {k: [] for k, v in batch.items()}
                for k, v in empty_batch.items():
                    empty_batch[k] = batch[k][i*self.concurrency: i*self.concurrency + self.concurrency]
                sample_groups.append(empty_batch)
            return sample_groups
        
    def get_model_config(self):
        config = {
            "batch_size": self.dataloader.batch_size,
            "stride": self.dataloader.stride,
            "max_length": self.dataloader.max_length,
            "train_fraction": self.dataloader.train_fraction,
            "use_large": self.dataloader.use_large,
            "concurrency": self.concurrency,
            "learning_rate": self.model_params.learning_rate,
            "epochs": self.epochs,
            "dropout": self.model_params.dropout,
            "save_all": self.model_params.save_all,
            "artefact_dir": self.model_params.artefact_dir,
            "clip_grad": self.clip_grad,
            "early_stopping_patience": self.early_stopping_patience
        }
        return config
    
    def cross_entropy_loss(self, logits, labels, attention_mask, size_average=True):
        print(logits.shape, labels.shape)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
        if attention_mask is not None:
            loss = loss * attention_mask.view(-1)
        if size_average:
            loss = loss.sum() / attention_mask.sum()
        print(loss)
        return loss

    def train(self):
        self.model_params.model.to(self.device)
        self.model_params.start_time = datetime.datetime.now()

        print(f"TRAINING START TIME: {str(self.model_params.start_time)}", flush=True)
        print("="*40, flush=True)
        print("="*40, flush=True)
        for epoch in range(self.model_params.epoch_num, self.epochs):
            
            #####################################################
            # TRAIN
            #####################################################

            self.model_params.model.train()
            self.epoch_params.reset_epoch()
            # self.model_params.epoch_list.append(epoch)
            print("#"*40, flush=True)
            print("#"*40, flush=True)
            print(
                f"EPOCH START TIME: {str(self.epoch_params.epoch_time)}\tEPOCH NUMBER: {epoch}", flush=True)
            for batch_idx, batch in enumerate(self.dataloader.train_dataloader):
                self.epoch_params.reset_batch()
                print("$"*40, flush=True)
                print(
                    f"BATCH START TIME: {str(self.epoch_params.batch_time)}\tBATCH NUMBER: {batch_idx}", flush=True)
                
                # zero the parameter gradients
                self.model_params.optimizer.zero_grad()
                sample_groups = self.group_samples(batch)
                loss = None
                # Take sub batches, accumulate gradients and loss
                for group in sample_groups:
                    for k, v in group.items():
                        # remove unwanted index
                        if k in ["idx"]:
                            continue
                        # put on device
                        group[k] = v.to(self.device)
                    del group["idx"]
                    # del _temp
                    # forward + backward + optimize
                    # group = torch.tensor(**group).to(torch.long64)
                    try:
                        outputs = self.model_params.model(**group)
                    except Exception as e:
                        raise
                    print(outputs.logits.shape, group["labels"].shape, group["attention_mask"].shape)
                        
                    self.epoch_params.batch_predictions.extend(outputs.logits.argmax(dim=2).detach())
                    self.epoch_params.batch_labels.extend(group['labels'].detach())
                    self.epoch_params.batch_attention.extend(group['attention_mask'].detach())

                loss = self.cross_entropy_loss(
                    logits=torch.cat(self.epoch_params.batch_predictions,dim=2).to(torch.float32).requires_grad_(),
                    labels=torch.cat(self.epoch_params.batch_labels,dim=2).to(torch.float32).requires_grad_(),
                    attention_mask=torch.cat(self.epoch_params.batch_attention,dim=2).to(torch.float32).requires_grad_()
                )
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_params.model.parameters(), self.clip_grad)
                self.model_params.optimizer.step()
                self.epoch_params.epoch_train_loss.append(loss.detach().item())
                print(f"Loss after {self.model_params.train_steps} steps: {loss.detach().item()}, | time elapsed: {datetime.datetime.now()-self.model_params.start_time}", flush=True)
                self.model_params.train_steps+=1
                
                true_predictions = [
                    [self.dataloader.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100] 
                    for prediction, label in zip(self.epoch_params.batch_predictions, self.epoch_params.batch_labels)
                ]
                true_labels = [
                    [self.dataloader.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(self.epoch_params.batch_predictions, self.epoch_params.batch_labels)
                ]
                self.model_params.train_metric.add_batch(predictions=true_predictions, references=true_labels)

                print(f"TOTAL BATCH TIME: {datetime.datetime.now()-self.epoch_params.batch_time}", flush=True)

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
                    sample_groups = self.group_samples(batch)
                    loss = None
                    # Take sub batches, accumulate gradients and loss
                    for group in sample_groups:
                        for k, v in group.items():
                            # remove unwanted index
                            if k in ["idx"]:
                                continue
                            # put on device
                            group[k] = v.to(self.device)
                        del group["idx"]

                        # del _temp
                        # forward + backward + optimize
                        # group = torch.tensor(**group).to(torch.long64)
                        try:
                            outputs = self.model_params.model(**group)
                        except Exception as e:
                            raise
                        self.epoch_params.batch_predictions.extend(outputs.logits.argmax(dim=2).detach())
                        self.epoch_params.batch_labels.extend(group['labels'].detach())
                        self.epoch_params.batch_attention.extend(group['attention_mask'].detach())
            
                    loss = self.cross_entropy_loss(
                        logits=torch.cat(self.epoch_params.batch_predictions,dim=3).to(torch.float32),
                        labels=torch.cat(self.epoch_params.batch_labels,dim=3).to(torch.float32),
                        attention_mask=torch.cat(self.epoch_params.batch_attention,dim=3).to(torch.float32)
                    )

                    self.epoch_params.epoch_eval_loss.append(loss.detach().item())
                    
                    true_predictions = [
                        [self.dataloader.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100] 
                        for prediction, label in zip(self.epoch_params.batch_predictions, self.epoch_params.batch_labels)
                    ]
                    true_labels = [
                        [self.dataloader.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(self.epoch_params.batch_predictions, self.epoch_params.batch_labels)
                    ]
                    self.model_params.eval_metric.add_batch(predictions=true_predictions, references=true_labels)

            eval_score = self.model_params.eval_metric.compute()
            print("EVAL SCORE = ", flush=True)
            for k, v in eval_score.items():
                print(k, ":", v, flush=True)
            self.model_params.eval_f1_scores.append(eval_score["overall_f1"])

            eval_loss = np.mean(self.epoch_params.epoch_eval_loss)
            print("EVAL LOSS = ", eval_loss, flush=True)
            self.model_params.eval_losses.append(eval_loss)
            
            if (eval_loss < self.model_params.last_best or eval_score["overall_f1"] > self.model_params.last_best_f1) or (self.model_params.save_all):
                self.model_params.last_best = min(self.model_params.last_best, eval_loss)
                self.model_params.last_best_f1 = max(self.model_params.last_best_f1, eval_score["overall_f1"])
                self.model_params.save_model(epoch, self.get_model_config(), self.dataloader)

            self.model_params.scheduler.step(eval_loss)
            if len(self.model_params.eval_losses) > self.early_stopping_patience and min(self.model_params.eval_losses[-1*self.early_stopping_patience:]) > self.model_params.last_best:
                print("EARLY STOPPING", flush=True)
                break

            print(f"TOTAL EPOCH TIME: {datetime.datetime.now()-self.epoch_params.epoch_time}", flush=True)
            print("#"*40, flush=True)
            print("#"*40, flush=True)

        self.model_params.end_time = datetime.datetime.now()
        self.model_params.total_time = self.model_params.end_time - self.model_params.start_time
        print(f"TOTAL TRAINING TIME: {self.model_params.total_time}", flush=True)
        print("="*40, flush=True)
        print("="*40, flush=True)
