import json
import torch
import mlflow
from transformers import AutoProcessor, AutoModelForTokenClassification
from transformers import AdamW
from datasets import load_metric

from mixins import TrainerMixin, TimeMixin


class BaseClass:
    def __init__(self, use_large):
        if use_large:
            self.model_name = "microsoft/layoutlmv2-large-uncased"
            self.processor_name = "microsoft/layoutlmv2-large-uncased"
        else:
            self.model_name = "microsoft/layoutlmv2-base-uncased"
            self.processor_name = "microsoft/layoutlmv2-base-uncased"


class Processor(BaseClass):
    def __init__(self, use_large):
        super().__init__(use_large)

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.processor_name, revision="no_ocr")


class Model(TrainerMixin, TimeMixin, BaseClass):
    def __init__(self,
                 dropout,
                 save_all,
                 learning_rate,
                 use_large,
                 optimizer=None,
                 epoch_num=0,
                 model=None
                 ):
        TrainerMixin.__init__(self)
        TimeMixin.__init__(self)
        BaseClass.__init__(self, use_large)
        self.dropout = dropout
        self.save_all = save_all
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.model = model

    def load_model(self, labels):
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(labels), ignore_mismatched_sizes=True, revision="no_ocr"
        )
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', verbose=True, patience=5)
        self.epoch_num = 0
        self.train_metric = load_metric("seqeval")
        self.eval_metric = load_metric("seqeval")
        

    def save_model(self, epoch, dataloader):
        metadata = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.eval_losses[-1],
            'labels': dataloader.labels,
            'label_to_ids': dataloader.label2id,
            'train_score': self.train_f1_scores[-1],
            'test_score': self.eval_f1_scores[-1]
        }

        mlflow.pytorch.log_model(
            artifact_path="models",
            pytorch_model=self.model,
            registered_model_name=f"best_{epoch}_{self.eval_losses[-1]:.4f}_{self.eval_f1_scores[-1]:.4f}.pt",
            extra_files={  # Attach the metadata as an extra file
                "metadata.json": json.dumps(metadata)
            }
        )
        self.saved_files.append(f"best_{epoch}_{self.eval_losses[-1]:.4f}_{self.eval_f1_scores[-1]:.4f}.pt")
