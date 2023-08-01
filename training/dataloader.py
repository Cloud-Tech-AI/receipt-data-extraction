import numpy as np


class ReceiptDataset():
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class ReceiptDataLoader():
    def __init__(self):
        pass

    @staticmethod
    def get_train_test_split(annotations, train_fraction):
        unique_docs = sorted(
            list(set([i["image_path"].split("/")[-1].split("-")[0] for i in annotations])))
        np.random.shuffle(unique_docs)
        train_docs = unique_docs[:int(len(unique_docs) * train_fraction)]
        test_docs = unique_docs[int(len(unique_docs) * train_fraction):]
        train_annotations = [
            i for i in annotations
            if i["image_path"].split("/")[-1].split("-")[0] in train_docs
        ]
        test_annotations = [
            i for i in annotations
            if i["image_path"].split("/")[-1].split("-")[0] in test_docs
        ]

        return train_annotations, test_annotations

    def get_dataset(self):
        pass
