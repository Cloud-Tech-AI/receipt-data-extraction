import os
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from transformers import AdamW

from mixins import TrainerMixin, TimeMixin


class BaseClass:
    def __init__(self, use_large=False):
        if use_large:
            self.model_name = "microsoft/layoutlmv2-large-uncased"
            self.processor_name = "microsoft/layoutlmv2-large-uncased"
        else:
            self.model_name = "microsoft/layoutlmv2-base-uncased"
            self.processor_name = "microsoft/layoutlmv2-base-uncased"


class Processor(BaseClass):
    def __init__(self, use_large=False):
        super().__init__(use_large)

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.processor_name, revision="no_ocr")


class Model(TrainerMixin, TimeMixin, BaseClass):
    def __init__(self,
                 dropout,
                 save_all,
                 artifect_dir,
                 learning_rate,
                 use_large=False,
                 optimizer=None,
                 epoch_num=0,
                 model=None
                 ):
        super().__init__(use_large)
        self.dropout = dropout
        self.save_all = save_all
        self.artifect_dir = artifect_dir
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
            self.model.optimizer, 'min', verbose=True, patience=5)
        self.epoch_num = 0

    def save_model(self, epoch, config, dataloader):
        if "saved_models" not in list(os.listdir(f"{self.artefact_dir}")):
            os.mkdir(os.path.join(self.artefact_dir, "saved_models"))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.eval_losses[-1],
                'labels': dataloader.labels,
                'label_to_ids': dataloader.label2id,
                'config': config,
                'train_score': self.train_f1_scores[-1],
                'test_score': self.eval_f1_scores[-1]
            },
            os.path.join(
                self.artefact_dir, "saved_models",
                f"best_{epoch}_{self.eval_losses[-1]:.4f}_{self.eval_f1_scores[-1]:.4f}.pt"
            )
        )

        self.saved_files.append(
            os.path.join(
                self.artefact_dir, "saved_models",
                f"best_{epoch}_{self.eval_losses[-1]:.4f}_{self.eval_f1_scores[-1]:.4f}.pt"
            )
        )
