import io
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    BatchSampler,
    RandomSampler,
)
import boto3
from PIL import Image

from model import Processor


@dataclass
class ReceiptDataset(Dataset):
    annotations: List[Dict]
    stride: int
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    use_large: bool
    max_length: int
    bucket_name: Optional[str]
    split_name: str
    s3client = boto3.client("s3")

    def __post_init__(self) -> None:
        processor = Processor(use_large=self.use_large)
        processor.load_processor()
        self.processor = processor.processor

    def __len__(self):
        return len(self.annotations)

    def process_data(self, data):
        images = []
        for path in data["imgs"]:
            if self.bucket_name is None:
                images.append(Image.open(path).convert("RGB"))
            else:
                images.append(
                    Image.open(
                        io.BytesIO(
                            self.s3client.get_object(Bucket=self.bucket_name, Key=path)[
                                "Body"
                            ].read()
                        )
                    ).convert("RGB")
                )
        words = data["words"]
        boxes = data["boxes"]
        labels = data["labels"]
        assert len(images) == len(words) == len(boxes) == len(labels)

        word_labels = [
            [self.label2id[single_label] for label in labels for single_label in label]
        ]

        encoded_inputs = self.processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self.stride,
        )
        if isinstance(encoded_inputs["image"], list):
            encoded_inputs["image"] = torch.stack(encoded_inputs["image"]).repeat(
                1, 1, 1, 1
            )

        return encoded_inputs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        ret_obj = {
            "input_ids": torch.tensor([]).type(torch.LongTensor),
            "token_type_ids": torch.tensor([]).type(torch.LongTensor),
            "attention_mask": torch.tensor([]).type(torch.LongTensor),
            "bbox": torch.tensor([]).type(torch.LongTensor),
            "labels": torch.tensor([]).type(torch.LongTensor),
            "image": torch.tensor([]).type(torch.LongTensor),
            "idx": torch.tensor([]).type(torch.LongTensor),
        }

        processed_dict = {
            i: self.process_data({k: [v] for k, v in self.annotations[i].items()})
            for i in idx
        }
        for idx, data in processed_dict.items():
            for key in ret_obj:
                if key == "idx":
                    ret_obj[key] = torch.cat([ret_obj[key], torch.tensor([idx])])
                    continue
                ret_obj[key] = torch.cat([ret_obj[key], data[key]])
                if ret_obj[key].dtype == torch.float32:
                    ret_obj[key] = ret_obj[key].type(torch.LongTensor)

        return ret_obj


@dataclass
class ReceiptDataLoader:
    batch_size: int
    stride: int
    max_length: int
    train_fraction: float
    use_large: bool
    data_path: Optional[str]
    bucket_name: Optional[str]
    s3client = boto3.client("s3")

    def __post_init__(self) -> None:
        self.annotations = self.load_data()
        self.labels = self.get_unique_labels()
        self.label2id = {v: k for k, v in enumerate(self.labels)}
        self.id2label = {k: v for k, v in enumerate(self.labels)}
        self.train_annotations, self.test_annotations = self.get_train_test_split()
        self.train_dataset, self.test_dataset = self.get_dataset()
        self.train_dataloader, self.test_dataloader = self.get_dataloader()

    def load_data(self):
        """Load processed data from S3 bucket or local file"""
        return (
            json.loads(
                self.s3client.get_object(
                    Bucket=self.bucket_name,
                    Key="preprocessed-data-train/processed_data.json",
                )["Body"].read()
            )
            if not self.data_path
            else json.loads(open(self.data_path, "r").read())
        )

    def get_unique_labels(self) -> List[str]:
        """Get unique labels from the dataset"""
        labels = []
        for annot in self.annotations:
            labels = list(set(labels + annot["labels"]))
        return labels

    def get_train_test_split(self) -> Tuple[List[Dict], List[Dict]]:
        """Get train and test annotations from annotations"""
        unique_docs = sorted(
            list(set([i["imgs"].split("/")[-1] for i in self.annotations]))
        )
        np.random.shuffle(unique_docs)
        train_docs = unique_docs[: int(len(unique_docs) * self.train_fraction)]
        test_docs = unique_docs[int(len(unique_docs) * self.train_fraction) :]
        train_annotations = [
            i for i in self.annotations if i["imgs"].split("/")[-1] in train_docs
        ]
        test_annotations = [
            i for i in self.annotations if i["imgs"].split("/")[-1] in test_docs
        ]

        return train_annotations, test_annotations

    def get_dataset(self):
        train_dataset = ReceiptDataset(
            self.train_annotations,
            self.stride,
            self.label2id,
            self.id2label,
            self.use_large,
            self.max_length,
            self.bucket_name,
            split_name="train",
        )
        test_dataset = ReceiptDataset(
            self.test_annotations,
            self.stride,
            self.label2id,
            self.id2label,
            self.use_large,
            self.max_length,
            self.bucket_name,
            split_name="val",
        )
        return train_dataset, test_dataset

    def get_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset, replacement=False)

        train_sampler = BatchSampler(
            train_sampler, batch_size=self.batch_size, drop_last=False
        )
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=None, num_workers=2
        )

        test_sampler = SequentialSampler(self.test_dataset)
        test_sampler = BatchSampler(
            test_sampler, batch_size=self.batch_size, drop_last=False
        )
        test_dataloader = DataLoader(
            self.test_dataset, sampler=test_sampler, batch_size=None, num_workers=2
        )

        return train_dataloader, test_dataloader
