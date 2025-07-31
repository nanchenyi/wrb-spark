import torch
from torch.utils.data import Dataset


class GLUEDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]
        self.token_type_ids = dataset["token_type_ids"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "token_type_ids": torch.tensor(self.token_type_ids[idx]),
            "labels": torch.tensor(self.labels[idx])
        }