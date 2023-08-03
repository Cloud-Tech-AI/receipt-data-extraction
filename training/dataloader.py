import json
import numpy as np
import torch
from PIL import Image
import pickle

from model import Model


class ReceiptDataset(torch.utils.data.Dataset):
    def __init__(self,
                 annotations,
                 stride,
                 label2id,
                 id2label,
                 use_large,
                 max_length,
                 split_name):
        self.annotations = annotations
        self.stride = stride
        self.label2id = label2id
        self.id2label = id2label
        self.split_name = split_name
        self.max_length = max_length
        model = Model(use_large=use_large)
        self.processor = model.load_processor()

    def __len__(self):
        return len(self.annotations)
    
    def preprocess_data(self, data):
        images = [Image.open(path).convert("RGB") for path in data['imgs']]
        words = data['words']
        boxes = data['boxes']
        labels = data['labels']
        assert len(images)==len(words)==len(boxes)==len(labels)

        word_labels = [[self.label2id[label] for label in labels[idx]] for idx,_ in enumerate(labels)]

        encoded_inputs = self.processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self.stride,
        )
        
        with open('encoded_inputs.json', 'w') as f:
            f.write(str(encoded_inputs))
        
        
        return encoded_inputs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        ret_obj = {
            'input_ids': torch.tensor([]).type(torch.LongTensor),
            'token_type_ids': torch.tensor([]).type(torch.LongTensor),
            'attention_mask': torch.tensor([]).type(torch.LongTensor),
            'bbox': torch.tensor([]).type(torch.LongTensor),
            'labels': torch.tensor([]).type(torch.LongTensor),
            'image': torch.tensor([]).type(torch.LongTensor),
            'idx': torch.tensor([]).type(torch.LongTensor)
        }
        processed_dict = {i:self.preprocess_data({k:[v] for k,v in self.annotations[i].items()}) for i in idx}
        return ret_obj

class ReceiptDataLoader:
    def __init__(self,
                data_path,
                 batch_size,
                stride,
                max_length,
                train_fraction,
                use_large
                 ):
        self.batch_size = batch_size
        self.stride = stride
        self.max_length = max_length
        self.train_fraction = train_fraction
        self.use_large = use_large
        self.annotations = json.loads(open(data_path, "r").read())
        self.labels = []
        for annot in self.annotations:
            self.labels = list(set(self.labels + annot["labels"]))
        self.label2id = {v: k for k, v in enumerate(self.labels)}
        self.id2label = {k: v for k, v in enumerate(self.labels)}
        self.annotations = self.annotations[:10]
        self.train_annotations, self.test_annotations = self.get_train_test_split()
        self.train_dataset, self.test_dataset = self.get_dataset()
        self.train_dataloader, self.test_dataloader = self.get_dataloader()

    def get_train_test_split(self):
        unique_docs = sorted(
            list(set([i["imgs"].split("/")[-1] for i in self.annotations])))
        np.random.shuffle(unique_docs)
        train_docs = unique_docs[:int(len(unique_docs) * self.train_fraction)]
        test_docs = unique_docs[int(len(unique_docs) * self.train_fraction):]
        train_annotations = [
            i for i in self.annotations
            if i["imgs"].split("/")[-1] in train_docs
        ]
        test_annotations = [
            i for i in self.annotations
            if i["imgs"].split("/")[-1] in test_docs
        ]

        return train_annotations, test_annotations

    def get_dataset(self):
        train_dataset = ReceiptDataset(self.train_annotations,
                                        self.stride,
                                        self.label2id,
                                        self.id2label,
                                        self.use_large,
                                        self.max_length,
                                        split_name='train')
        test_dataset = ReceiptDataset(self.test_annotations,
                                    self.stride,
                                    self.label2id,
                                    self.id2label,
                                    self.use_large,
                                    self.max_length,
                                    split_name='val')
        return train_dataset, test_dataset
    
    def get_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset,
                                                    replacement=False)
        
        train_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                    batch_size=self.batch_size,
                                                    drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                    sampler=train_sampler,
                                                    batch_size=None,
                                                    num_workers=4)
        
        test_sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        test_sampler = torch.utils.data.BatchSampler(test_sampler,
                                                    batch_size=self.batch_size,
                                                    drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    sampler=test_sampler,
                                                    batch_size=None,
                                                    num_workers=4)

        return train_dataloader, test_dataloader

